/**
 * Graph Analytics Service
 * Analyzes transaction networks and identifies fraud rings
 */

import express, { Application, Request, Response } from 'express';
import { Neo4jGraphDB } from './database/neo4j-client';
import { GraphAlgorithms } from './algorithms/graph-algorithms';
import { FraudRingDetector } from './detectors/fraud-ring-detector';
import { NetworkAnalyzer } from './analyzers/network-analyzer';
import { CommunityDetector } from './detectors/community-detector';
import winston from 'winston';
import prometheus from 'prom-client';

// Configuration
const config = {
  port: process.env.PORT || 3011,
  neo4j: {
    uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
    user: process.env.NEO4J_USER || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'password'
  },
  analysis: {
    maxDepth: 5,
    minClusterSize: 3,
    similarityThreshold: 0.7
  }
};

// Logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'graph-analytics.log' })
  ]
});

// Prometheus metrics
const register = new prometheus.Register();
const analysisCounter = new prometheus.Counter({
  name: 'graph_analysis_total',
  help: 'Total number of graph analyses',
  labelNames: ['type'],
  registers: [register]
});

const analysisDuration = new prometheus.Histogram({
  name: 'graph_analysis_duration_seconds',
  help: 'Duration of graph analysis operations',
  labelNames: ['type'],
  registers: [register]
});

// Express app
const app: Application = express();
app.use(express.json());

// Types
interface Node {
  id: string;
  type: 'user' | 'merchant' | 'device' | 'ip_address';
  properties: Record<string, any>;
}

interface Edge {
  source: string;
  target: string;
  type: string;
  weight: number;
  properties: Record<string, any>;
}

interface Graph {
  nodes: Node[];
  edges: Edge[];
}

interface FraudRing {
  id: string;
  members: string[];
  connections: Edge[];
  riskScore: number;
  patterns: string[];
  size: number;
}

interface NetworkMetrics {
  nodeId: string;
  degree: number;
  betweenness: number;
  closeness: number;
  pageRank: number;
  clusteringCoefficient: number;
  communityId?: string;
}

interface PathAnalysis {
  source: string;
  target: string;
  paths: Path[];
  shortestPathLength: number;
  totalPaths: number;
}

interface Path {
  nodes: string[];
  edges: Edge[];
  length: number;
  weight: number;
}

/**
 * Graph Database Client
 */
class GraphDatabaseClient {
  private db: Neo4jGraphDB;

  constructor() {
    this.db = new Neo4jGraphDB(config.neo4j);
  }

  async connect(): Promise<void> {
    await this.db.connect();
    logger.info('Connected to Neo4j');
  }

  async createNode(node: Node): Promise<void> {
    const query = `
      CREATE (n:${node.type} {id: $id})
      SET n += $properties
      RETURN n
    `;

    await this.db.run(query, {
      id: node.id,
      properties: node.properties
    });
  }

  async createEdge(edge: Edge): Promise<void> {
    const query = `
      MATCH (a {id: $source})
      MATCH (b {id: $target})
      CREATE (a)-[r:${edge.type} {weight: $weight}]->(b)
      SET r += $properties
      RETURN r
    `;

    await this.db.run(query, {
      source: edge.source,
      target: edge.target,
      weight: edge.weight,
      properties: edge.properties
    });
  }

  async getSubgraph(nodeId: string, depth: number = 2): Promise<Graph> {
    const query = `
      MATCH path = (start {id: $nodeId})-[*1..${depth}]-(connected)
      WITH nodes(path) as pathNodes, relationships(path) as pathRels
      UNWIND pathNodes as node
      WITH collect(DISTINCT node) as allNodes, pathRels
      UNWIND pathRels as rel
      WITH allNodes, collect(DISTINCT rel) as allRels
      RETURN allNodes, allRels
    `;

    const result = await this.db.run(query, { nodeId });

    const nodes: Node[] = result.records[0]?.get('allNodes').map((n: any) => ({
      id: n.properties.id,
      type: n.labels[0].toLowerCase(),
      properties: n.properties
    })) || [];

    const edges: Edge[] = result.records[0]?.get('allRels').map((r: any) => ({
      source: r.start.properties.id,
      target: r.end.properties.id,
      type: r.type,
      weight: r.properties.weight || 1,
      properties: r.properties
    })) || [];

    return { nodes, edges };
  }

  async findShortestPath(sourceId: string, targetId: string): Promise<Path | null> {
    const query = `
      MATCH (source {id: $sourceId}), (target {id: $targetId})
      MATCH path = shortestPath((source)-[*]-(target))
      RETURN path
    `;

    const result = await this.db.run(query, { sourceId, targetId });

    if (result.records.length === 0) {
      return null;
    }

    const path = result.records[0].get('path');
    const nodes = path.segments.map((s: any) => s.start.properties.id);
    nodes.push(path.segments[path.segments.length - 1].end.properties.id);

    const edges = path.segments.map((s: any) => ({
      source: s.start.properties.id,
      target: s.end.properties.id,
      type: s.relationship.type,
      weight: s.relationship.properties.weight || 1,
      properties: s.relationship.properties
    }));

    return {
      nodes,
      edges,
      length: path.length,
      weight: edges.reduce((sum, e) => sum + e.weight, 0)
    };
  }

  async getNodeMetrics(nodeId: string): Promise<NetworkMetrics> {
    // Degree centrality
    const degreeQuery = `
      MATCH (n {id: $nodeId})-[r]-()
      RETURN count(r) as degree
    `;
    const degreeResult = await this.db.run(degreeQuery, { nodeId });
    const degree = degreeResult.records[0]?.get('degree').toNumber() || 0;

    // PageRank
    const pageRankQuery = `
      CALL gds.pageRank.stream({
        nodeQuery: 'MATCH (n) RETURN id(n) AS id',
        relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target',
        maxIterations: 20,
        dampingFactor: 0.85
      })
      YIELD nodeId, score
      WITH gds.util.asNode(nodeId) AS node, score
      WHERE node.id = $nodeId
      RETURN score as pageRank
    `;
    const pageRankResult = await this.db.run(pageRankQuery, { nodeId });
    const pageRank = pageRankResult.records[0]?.get('pageRank') || 0;

    // Clustering coefficient
    const clusteringQuery = `
      MATCH (n {id: $nodeId})--(neighbor)
      WITH n, collect(DISTINCT neighbor) as neighbors
      UNWIND neighbors as n1
      UNWIND neighbors as n2
      WITH n, n1, n2, neighbors
      WHERE id(n1) < id(n2)
      OPTIONAL MATCH (n1)--(n2)
      WITH n, count(DISTINCT n1) as possibleConnections, 
           count(DISTINCT CASE WHEN (n1)--(n2) THEN 1 END) as actualConnections
      RETURN CASE 
        WHEN possibleConnections = 0 THEN 0 
        ELSE toFloat(actualConnections) / possibleConnections 
      END as clustering
    `;
    const clusteringResult = await this.db.run(clusteringQuery, { nodeId });
    const clusteringCoefficient = clusteringResult.records[0]?.get('clustering') || 0;

    return {
      nodeId,
      degree,
      betweenness: 0, // Placeholder
      closeness: 0, // Placeholder
      pageRank,
      clusteringCoefficient
    };
  }

  async close(): Promise<void> {
    await this.db.close();
  }
}

/**
 * Fraud Ring Detection Service
 */
class FraudRingDetectionService {
  private db: GraphDatabaseClient;
  private detector: FraudRingDetector;

  constructor(db: GraphDatabaseClient) {
    this.db = db;
    this.detector = new FraudRingDetector();
  }

  async detectFraudRings(): Promise<FraudRing[]> {
    const startTime = Date.now();

    try {
      // Find densely connected components
      const communities = await this.findDenseCommunities();

      // Analyze each community for fraud patterns
      const fraudRings: FraudRing[] = [];

      for (const community of communities) {
        const ring = await this.analyzeCommunity(community);
        if (ring && ring.riskScore > 70) {
          fraudRings.push(ring);
        }
      }

      analysisCounter.labels('fraud_ring_detection').inc();
      analysisDuration.labels('fraud_ring_detection').observe((Date.now() - startTime) / 1000);

      return fraudRings;
    } catch (error) {
      logger.error('Error detecting fraud rings', { error });
      throw error;
    }
  }

  private async findDenseCommunities(): Promise<string[][]> {
    // Use Louvain algorithm for community detection
    const query = `
      CALL gds.louvain.stream({
        nodeQuery: 'MATCH (n) RETURN id(n) AS id',
        relationshipQuery: 'MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, r.weight AS weight'
      })
      YIELD nodeId, communityId
      WITH communityId, collect(gds.util.asNode(nodeId).id) as members
      WHERE size(members) >= ${config.analysis.minClusterSize}
      RETURN communityId, members
    `;

    const result = await this.db['db'].run(query, {});

    return result.records.map(record => record.get('members'));
  }

  private async analyzeCommunity(members: string[]): Promise<FraudRing | null> {
    // Get subgraph for community
    const subgraphs = await Promise.all(
      members.map(id => this.db.getSubgraph(id, 1))
    );

    // Merge subgraphs
    const allNodes = new Map<string, Node>();
    const allEdges: Edge[] = [];

    for (const subgraph of subgraphs) {
      subgraph.nodes.forEach(node => allNodes.set(node.id, node));
      allEdges.push(...subgraph.edges);
    }

    // Detect fraud patterns
    const patterns = this.detector.detectPatterns({
      nodes: Array.from(allNodes.values()),
      edges: allEdges
    });

    if (patterns.length === 0) {
      return null;
    }

    // Calculate risk score
    const riskScore = this.calculateRingRiskScore(patterns, allEdges.length, members.length);

    return {
      id: `ring_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      members,
      connections: allEdges,
      riskScore,
      patterns,
      size: members.length
    };
  }

  private calculateRingRiskScore(patterns: string[], edgeCount: number, memberCount: number): number {
    let score = 0;

    // Pattern-based scoring
    score += patterns.length * 15;

    // Density scoring
    const maxPossibleEdges = (memberCount * (memberCount - 1)) / 2;
    const density = edgeCount / maxPossibleEdges;
    score += density * 30;

    // Size scoring
    if (memberCount > 10) score += 20;
    else if (memberCount > 5) score += 10;

    return Math.min(score, 100);
  }
}

/**
 * Network Analysis Service
 */
class NetworkAnalysisService {
  private db: GraphDatabaseClient;
  private analyzer: NetworkAnalyzer;

  constructor(db: GraphDatabaseClient) {
    this.db = db;
    this.analyzer = new NetworkAnalyzer();
  }

  async analyzeNode(nodeId: string): Promise<any> {
    const startTime = Date.now();

    try {
      // Get node metrics
      const metrics = await this.db.getNodeMetrics(nodeId);

      // Get subgraph
      const subgraph = await this.db.getSubgraph(nodeId, config.analysis.maxDepth);

      // Analyze network structure
      const structureAnalysis = this.analyzer.analyzeStructure(subgraph);

      // Identify anomalies
      const anomalies = this.analyzer.detectAnomalies(metrics, structureAnalysis);

      analysisCounter.labels('node_analysis').inc();
      analysisDuration.labels('node_analysis').observe((Date.now() - startTime) / 1000);

      return {
        nodeId,
        metrics,
        structure: structureAnalysis,
        anomalies,
        riskScore: this.calculateNodeRiskScore(metrics, anomalies)
      };
    } catch (error) {
      logger.error('Error analyzing node', { error, nodeId });
      throw error;
    }
  }

  private calculateNodeRiskScore(metrics: NetworkMetrics, anomalies: string[]): number {
    let score = 0;

    // High degree centrality
    if (metrics.degree > 50) score += 20;
    else if (metrics.degree > 20) score += 10;

    // High PageRank
    if (metrics.pageRank > 0.01) score += 15;

    // Low clustering coefficient (unusual for legitimate users)
    if (metrics.clusteringCoefficient < 0.1) score += 15;

    // Anomaly-based scoring
    score += anomalies.length * 10;

    return Math.min(score, 100);
  }
}

/**
 * Path Analysis Service
 */
class PathAnalysisService {
  private db: GraphDatabaseClient;

  constructor(db: GraphDatabaseClient) {
    this.db = db;
  }

  async analyzePaths(sourceId: string, targetId: string): Promise<PathAnalysis> {
    const startTime = Date.now();

    try {
      // Find shortest path
      const shortestPath = await this.db.findShortestPath(sourceId, targetId);

      if (!shortestPath) {
        return {
          source: sourceId,
          target: targetId,
          paths: [],
          shortestPathLength: -1,
          totalPaths: 0
        };
      }

      // Find all paths (limited)
      const allPaths = await this.findAllPaths(sourceId, targetId, 5);

      analysisCounter.labels('path_analysis').inc();
      analysisDuration.labels('path_analysis').observe((Date.now() - startTime) / 1000);

      return {
        source: sourceId,
        target: targetId,
        paths: allPaths,
        shortestPathLength: shortestPath.length,
        totalPaths: allPaths.length
      };
    } catch (error) {
      logger.error('Error analyzing paths', { error, sourceId, targetId });
      throw error;
    }
  }

  private async findAllPaths(sourceId: string, targetId: string, maxLength: number): Promise<Path[]> {
    const query = `
      MATCH (source {id: $sourceId}), (target {id: $targetId})
      MATCH path = (source)-[*1..${maxLength}]-(target)
      RETURN path
      LIMIT 100
    `;

    const result = await this.db['db'].run(query, { sourceId, targetId });

    return result.records.map(record => {
      const path = record.get('path');
      const nodes = path.segments.map((s: any) => s.start.properties.id);
      nodes.push(path.segments[path.segments.length - 1].end.properties.id);

      const edges = path.segments.map((s: any) => ({
        source: s.start.properties.id,
        target: s.end.properties.id,
        type: s.relationship.type,
        weight: s.relationship.properties.weight || 1,
        properties: s.relationship.properties
      }));

      return {
        nodes,
        edges,
        length: path.length,
        weight: edges.reduce((sum, e) => sum + e.weight, 0)
      };
    });
  }
}

// Initialize services
const dbClient = new GraphDatabaseClient();
const fraudRingService = new FraudRingDetectionService(dbClient);
const networkAnalysisService = new NetworkAnalysisService(dbClient);
const pathAnalysisService = new PathAnalysisService(dbClient);

// Routes
app.post('/api/v1/graph/node', async (req: Request, res: Response) => {
  try {
    const node: Node = req.body;
    await dbClient.createNode(node);
    res.json({ success: true, nodeId: node.id });
  } catch (error) {
    logger.error('Error creating node', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/v1/graph/edge', async (req: Request, res: Response) => {
  try {
    const edge: Edge = req.body;
    await dbClient.createEdge(edge);
    res.json({ success: true });
  } catch (error) {
    logger.error('Error creating edge', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/graph/subgraph/:nodeId', async (req: Request, res: Response) => {
  try {
    const { nodeId } = req.params;
    const depth = parseInt(req.query.depth as string) || 2;
    const subgraph = await dbClient.getSubgraph(nodeId, depth);
    res.json(subgraph);
  } catch (error) {
    logger.error('Error getting subgraph', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/analysis/fraud-rings', async (req: Request, res: Response) => {
  try {
    const fraudRings = await fraudRingService.detectFraudRings();
    res.json({ fraudRings, count: fraudRings.length });
  } catch (error) {
    logger.error('Error detecting fraud rings', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/analysis/node/:nodeId', async (req: Request, res: Response) => {
  try {
    const { nodeId } = req.params;
    const analysis = await networkAnalysisService.analyzeNode(nodeId);
    res.json(analysis);
  } catch (error) {
    logger.error('Error analyzing node', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/analysis/path', async (req: Request, res: Response) => {
  try {
    const { source, target } = req.query;
    if (!source || !target) {
      return res.status(400).json({ error: 'Source and target required' });
    }
    const analysis = await pathAnalysisService.analyzePaths(source as string, target as string);
    res.json(analysis);
  } catch (error) {
    logger.error('Error analyzing path', { error });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/metrics', async (req: Request, res: Response) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'healthy', timestamp: Date.now() });
});

// Start server
async function start() {
  try {
    await dbClient.connect();

    app.listen(config.port, () => {
      logger.info(`Graph Analytics Service started on port ${config.port}`);
    });
  } catch (error) {
    logger.error('Failed to start service', { error });
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await dbClient.close();
  process.exit(0);
});

start();

export { GraphDatabaseClient, FraudRingDetectionService, NetworkAnalysisService, PathAnalysisService };

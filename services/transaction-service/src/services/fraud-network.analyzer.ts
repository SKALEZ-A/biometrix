import { Driver, Session } from 'neo4j-driver';
import { txnDbManager } from '../config/database.config';

export interface FraudNetworkNode {
  id: string;
  type: 'user' | 'merchant' | 'device' | 'ip_address' | 'card';
  properties: Record<string, any>;
  riskScore: number;
  fraudCount: number;
}

export interface FraudNetworkEdge {
  from: string;
  to: string;
  type: 'transacted_with' | 'used_device' | 'shared_ip' | 'shared_card' | 'linked_to';
  weight: number;
  timestamp: number;
  properties: Record<string, any>;
}

export interface FraudNetwork {
  nodes: FraudNetworkNode[];
  edges: FraudNetworkEdge[];
  clusters: FraudCluster[];
  riskScore: number;
  suspiciousPatterns: string[];
}

export interface FraudCluster {
  clusterId: string;
  nodes: string[];
  clusterType: 'fraud_ring' | 'money_mule' | 'account_takeover' | 'synthetic_identity';
  riskScore: number;
  transactionCount: number;
  totalAmount: number;
}

export class FraudNetworkAnalyzer {
  private driver: Driver;

  constructor() {
    this.driver = txnDbManager.getNeo4jDriver();
  }

  async updateTransactionGraph(params: {
    userId: string;
    merchantId: string;
    transactionId: string;
    riskScore: number;
    amount: number;
    timestamp: number;
  }): Promise<void> {
    const session = this.driver.session();

    try {
      await session.run(
        `
        MERGE (u:User {id: $userId})
        ON CREATE SET u.createdAt = timestamp(), u.fraudCount = 0, u.totalTransactions = 0
        ON MATCH SET u.totalTransactions = u.totalTransactions + 1
        
        MERGE (m:Merchant {id: $merchantId})
        ON CREATE SET m.createdAt = timestamp(), m.fraudCount = 0, m.totalTransactions = 0
        ON MATCH SET m.totalTransactions = m.totalTransactions + 1
        
        CREATE (t:Transaction {
          id: $transactionId,
          amount: $amount,
          riskScore: $riskScore,
          timestamp: $timestamp
        })
        
        CREATE (u)-[:MADE_TRANSACTION {
          timestamp: $timestamp,
          amount: $amount,
          riskScore: $riskScore
        }]->(t)
        
        CREATE (t)-[:AT_MERCHANT {
          timestamp: $timestamp
        }]->(m)
        
        WITH u, m
        MERGE (u)-[r:TRANSACTED_WITH]->(m)
        ON CREATE SET r.count = 1, r.totalAmount = $amount, r.firstTransaction = $timestamp
        ON MATCH SET r.count = r.count + 1, r.totalAmount = r.totalAmount + $amount, r.lastTransaction = $timestamp
        `,
        params
      );

      // Update fraud count if high risk
      if (params.riskScore > 70) {
        await session.run(
          `
          MATCH (u:User {id: $userId})
          SET u.fraudCount = u.fraudCount + 1
          
          MATCH (m:Merchant {id: $merchantId})
          SET m.fraudCount = m.fraudCount + 1
          `,
          { userId: params.userId, merchantId: params.merchantId }
        );
      }
    } finally {
      await session.close();
    }
  }

  async detectFraudNetwork(userId: string, depth: number = 2): Promise<FraudNetwork> {
    const session = this.driver.session();

    try {
      // Find connected nodes
      const nodesResult = await session.run(
        `
        MATCH path = (u:User {id: $userId})-[*1..${depth}]-(connected)
        WHERE connected:User OR connected:Merchant OR connected:Device
        WITH DISTINCT connected
        RETURN 
          connected.id as id,
          labels(connected)[0] as type,
          properties(connected) as properties
        `,
        { userId }
      );

      const nodes: FraudNetworkNode[] = nodesResult.records.map((record) => ({
        id: record.get('id'),
        type: record.get('type').toLowerCase(),
        properties: record.get('properties'),
        riskScore: this.calculateNodeRiskScore(record.get('properties')),
        fraudCount: record.get('properties').fraudCount || 0,
      }));

      // Find edges between nodes
      const nodeIds = nodes.map((n) => n.id);
      const edgesResult = await session.run(
        `
        MATCH (a)-[r]->(b)
        WHERE a.id IN $nodeIds AND b.id IN $nodeIds
        RETURN 
          a.id as from,
          b.id as to,
          type(r) as type,
          properties(r) as properties
        `,
        { nodeIds }
      );

      const edges: FraudNetworkEdge[] = edgesResult.records.map((record) => ({
        from: record.get('from'),
        to: record.get('to'),
        type: this.normalizeEdgeType(record.get('type')),
        weight: record.get('properties').count || 1,
        timestamp: record.get('properties').lastTransaction || Date.now(),
        properties: record.get('properties'),
      }));

      // Detect fraud clusters
      const clusters = await this.detectFraudClusters(session, nodeIds);

      // Identify suspicious patterns
      const suspiciousPatterns = this.identifySuspiciousPatterns(nodes, edges, clusters);

      // Calculate overall network risk score
      const riskScore = this.calculateNetworkRiskScore(nodes, edges, clusters);

      return {
        nodes,
        edges,
        clusters,
        riskScore,
        suspiciousPatterns,
      };
    } finally {
      await session.close();
    }
  }

  async detectFraudRings(minClusterSize: number = 3): Promise<FraudCluster[]> {
    const session = this.driver.session();

    try {
      const result = await session.run(
        `
        CALL gds.louvain.stream('fraud-network')
        YIELD nodeId, communityId
        WITH communityId, collect(gds.util.asNode(nodeId).id) as members
        WHERE size(members) >= $minClusterSize
        
        MATCH (u:User)
        WHERE u.id IN members
        WITH communityId, members, 
             sum(u.fraudCount) as totalFraud,
             sum(u.totalTransactions) as totalTransactions
        
        RETURN 
          communityId,
          members,
          totalFraud,
          totalTransactions,
          toFloat(totalFraud) / totalTransactions as fraudRate
        ORDER BY fraudRate DESC
        `,
        { minClusterSize }
      );

      return result.records.map((record) => ({
        clusterId: `cluster_${record.get('communityId')}`,
        nodes: record.get('members'),
        clusterType: this.determineClusterType(record.get('fraudRate')),
        riskScore: record.get('fraudRate') * 100,
        transactionCount: record.get('totalTransactions'),
        totalAmount: 0, // Would need additional query
      }));
    } catch (error) {
      console.error('Error detecting fraud rings:', error);
      return [];
    } finally {
      await session.close();
    }
  }

  async findMoneyMules(userId: string): Promise<string[]> {
    const session = this.driver.session();

    try {
      const result = await session.run(
        `
        MATCH (u:User {id: $userId})-[:TRANSACTED_WITH*2..3]-(mule:User)
        WHERE mule.id <> $userId
        WITH mule, count(*) as connectionCount
        WHERE connectionCount > 5
        
        MATCH (mule)-[t:MADE_TRANSACTION]->()
        WITH mule, 
             count(t) as totalTransactions,
             sum(t.amount) as totalAmount,
             avg(t.amount) as avgAmount,
             stdDev(t.amount) as stdDevAmount
        
        WHERE totalTransactions > 20 
          AND stdDevAmount / avgAmount > 0.5
          AND mule.fraudCount > 2
        
        RETURN mule.id as muleId
        ORDER BY totalTransactions DESC
        LIMIT 10
        `,
        { userId }
      );

      return result.records.map((record) => record.get('muleId'));
    } finally {
      await session.close();
    }
  }

  async detectAccountTakeover(userId: string): Promise<{
    isLikelyTakeover: boolean;
    confidence: number;
    indicators: string[];
  }> {
    const session = this.driver.session();

    try {
      const result = await session.run(
        `
        MATCH (u:User {id: $userId})-[t:MADE_TRANSACTION]->(txn:Transaction)
        WITH u, txn
        ORDER BY txn.timestamp DESC
        LIMIT 50
        
        WITH u,
             collect(txn.timestamp) as timestamps,
             collect(txn.amount) as amounts,
             collect(txn.riskScore) as riskScores
        
        RETURN 
          u.id as userId,
          timestamps,
          amounts,
          riskScores,
          u.fraudCount as fraudCount
        `,
        { userId }
      );

      if (result.records.length === 0) {
        return {
          isLikelyTakeover: false,
          confidence: 0,
          indicators: [],
        };
      }

      const record = result.records[0];
      const timestamps = record.get('timestamps');
      const amounts = record.get('amounts');
      const riskScores = record.get('riskScores');
      const fraudCount = record.get('fraudCount');

      const indicators: string[] = [];
      let confidence = 0;

      // Check for sudden spike in transaction frequency
      if (timestamps.length >= 10) {
        const recentInterval = timestamps[0] - timestamps[9];
        const olderInterval = timestamps.length > 20 ? timestamps[10] - timestamps[19] : recentInterval;
        
        if (recentInterval < olderInterval * 0.3) {
          indicators.push('Sudden increase in transaction frequency');
          confidence += 0.25;
        }
      }

      // Check for unusual transaction amounts
      const avgAmount = amounts.reduce((sum: number, a: number) => sum + a, 0) / amounts.length;
      const recentAvg = amounts.slice(0, 5).reduce((sum: number, a: number) => sum + a, 0) / 5;
      
      if (recentAvg > avgAmount * 2) {
        indicators.push('Unusual transaction amounts');
        confidence += 0.25;
      }

      // Check for elevated risk scores
      const avgRiskScore = riskScores.reduce((sum: number, r: number) => sum + r, 0) / riskScores.length;
      const recentRiskAvg = riskScores.slice(0, 5).reduce((sum: number, r: number) => sum + r, 0) / 5;
      
      if (recentRiskAvg > avgRiskScore * 1.5 && recentRiskAvg > 60) {
        indicators.push('Elevated risk scores in recent transactions');
        confidence += 0.3;
      }

      // Check fraud history
      if (fraudCount > 0) {
        indicators.push('Previous fraud incidents');
        confidence += 0.2;
      }

      return {
        isLikelyTakeover: confidence > 0.5,
        confidence: Math.min(confidence, 1.0),
        indicators,
      };
    } finally {
      await session.close();
    }
  }

  async linkDeviceToUser(userId: string, deviceId: string, timestamp: number): Promise<void> {
    const session = this.driver.session();

    try {
      await session.run(
        `
        MERGE (u:User {id: $userId})
        MERGE (d:Device {id: $deviceId})
        ON CREATE SET d.createdAt = timestamp(), d.userCount = 0
        
        MERGE (u)-[r:USED_DEVICE]->(d)
        ON CREATE SET r.firstSeen = $timestamp, r.count = 1
        ON MATCH SET r.lastSeen = $timestamp, r.count = r.count + 1
        
        WITH d
        MATCH (d)<-[:USED_DEVICE]-(users:User)
        WITH d, count(DISTINCT users) as userCount
        SET d.userCount = userCount
        `,
        { userId, deviceId, timestamp }
      );
    } finally {
      await session.close();
    }
  }

  async findSharedDevices(userId: string): Promise<Array<{
    deviceId: string;
    sharedWith: string[];
    riskScore: number;
  }>> {
    const session = this.driver.session();

    try {
      const result = await session.run(
        `
        MATCH (u:User {id: $userId})-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(other:User)
        WHERE other.id <> $userId
        WITH d, collect(DISTINCT other.id) as sharedWith, d.userCount as userCount
        WHERE size(sharedWith) > 0
        RETURN 
          d.id as deviceId,
          sharedWith,
          userCount,
          CASE 
            WHEN userCount > 10 THEN 90
            WHEN userCount > 5 THEN 70
            WHEN userCount > 2 THEN 50
            ELSE 30
          END as riskScore
        ORDER BY riskScore DESC
        `,
        { userId }
      );

      return result.records.map((record) => ({
        deviceId: record.get('deviceId'),
        sharedWith: record.get('sharedWith'),
        riskScore: record.get('riskScore'),
      }));
    } finally {
      await session.close();
    }
  }

  private async detectFraudClusters(session: Session, nodeIds: string[]): Promise<FraudCluster[]> {
    // Simplified cluster detection using connected components
    const result = await session.run(
      `
      MATCH (n)
      WHERE n.id IN $nodeIds AND n.fraudCount > 0
      WITH collect(n.id) as fraudNodes
      
      UNWIND fraudNodes as nodeId
      MATCH path = (n {id: nodeId})-[*1..2]-(connected)
      WHERE connected.id IN fraudNodes
      WITH nodeId, collect(DISTINCT connected.id) as cluster
      RETURN nodeId, cluster
      `,
      { nodeIds }
    );

    const clusters: Map<string, Set<string>> = new Map();
    
    result.records.forEach((record) => {
      const nodeId = record.get('nodeId');
      const cluster = record.get('cluster');
      
      if (!clusters.has(nodeId)) {
        clusters.set(nodeId, new Set([nodeId, ...cluster]));
      }
    });

    return Array.from(clusters.values()).map((nodes, index) => ({
      clusterId: `cluster_${index}`,
      nodes: Array.from(nodes),
      clusterType: 'fraud_ring',
      riskScore: 75,
      transactionCount: 0,
      totalAmount: 0,
    }));
  }

  private identifySuspiciousPatterns(
    nodes: FraudNetworkNode[],
    edges: FraudNetworkEdge[],
    clusters: FraudCluster[]
  ): string[] {
    const patterns: string[] = [];

    // High fraud count nodes
    const highFraudNodes = nodes.filter((n) => n.fraudCount > 3);
    if (highFraudNodes.length > 0) {
      patterns.push(`${highFraudNodes.length} nodes with high fraud history`);
    }

    // Dense connections
    const avgDegree = edges.length / nodes.length;
    if (avgDegree > 5) {
      patterns.push('Unusually dense network connections');
    }

    // Large clusters
    const largeClusters = clusters.filter((c) => c.nodes.length > 5);
    if (largeClusters.length > 0) {
      patterns.push(`${largeClusters.length} large fraud clusters detected`);
    }

    // Rapid transactions
    const recentEdges = edges.filter((e) => Date.now() - e.timestamp < 24 * 60 * 60 * 1000);
    if (recentEdges.length > edges.length * 0.5) {
      patterns.push('High volume of recent transactions');
    }

    return patterns;
  }

  private calculateNodeRiskScore(properties: any): number {
    let score = 0;

    if (properties.fraudCount > 0) {
      score += Math.min(properties.fraudCount * 20, 60);
    }

    if (properties.totalTransactions > 0) {
      const fraudRate = properties.fraudCount / properties.totalTransactions;
      score += fraudRate * 40;
    }

    return Math.min(score, 100);
  }

  private calculateNetworkRiskScore(
    nodes: FraudNetworkNode[],
    edges: FraudNetworkEdge[],
    clusters: FraudCluster[]
  ): number {
    const avgNodeRisk = nodes.reduce((sum, n) => sum + n.riskScore, 0) / nodes.length;
    const clusterRisk = clusters.length > 0
      ? clusters.reduce((sum, c) => sum + c.riskScore, 0) / clusters.length
      : 0;
    const densityRisk = Math.min((edges.length / nodes.length) * 10, 30);

    return Math.min(avgNodeRisk * 0.4 + clusterRisk * 0.4 + densityRisk * 0.2, 100);
  }

  private normalizeEdgeType(type: string): FraudNetworkEdge['type'] {
    const typeMap: Record<string, FraudNetworkEdge['type']> = {
      'TRANSACTED_WITH': 'transacted_with',
      'USED_DEVICE': 'used_device',
      'SHARED_IP': 'shared_ip',
      'SHARED_CARD': 'shared_card',
      'LINKED_TO': 'linked_to',
    };

    return typeMap[type] || 'linked_to';
  }

  private determineClusterType(fraudRate: number): FraudCluster['clusterType'] {
    if (fraudRate > 0.7) return 'fraud_ring';
    if (fraudRate > 0.5) return 'money_mule';
    if (fraudRate > 0.3) return 'account_takeover';
    return 'synthetic_identity';
  }
}

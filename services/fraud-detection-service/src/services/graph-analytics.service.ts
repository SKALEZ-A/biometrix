import { FraudNetwork, GraphNode, GraphEdge, CommunityDetectionResult } from '@shared/types/fraud.types';
import { Logger } from '@shared/utils/logger';

export class GraphAnalyticsService {
  private readonly logger = new Logger('GraphAnalyticsService');
  private adjacencyList: Map<string, Set<string>> = new Map();
  private nodeAttributes: Map<string, any> = new Map();
  private edgeWeights: Map<string, number> = new Map();

  async buildFraudNetwork(transactions: any[]): Promise<FraudNetwork> {
    this.adjacencyList.clear();
    this.nodeAttributes.clear();
    this.edgeWeights.clear();

    for (const tx of transactions) {
      this.addNode(tx.senderId, { type: 'user', riskScore: tx.senderRisk });
      this.addNode(tx.receiverId, { type: 'user', riskScore: tx.receiverRisk });
      this.addEdge(tx.senderId, tx.receiverId, tx.amount);
    }

    const communities = await this.detectCommunities();
    const centralNodes = this.calculateCentrality();
    const suspiciousPatterns = await this.detectSuspiciousPatterns();

    return {
      nodes: Array.from(this.nodeAttributes.entries()).map(([id, attrs]) => ({
        id,
        ...attrs
      })),
      edges: this.getEdges(),
      communities,
      centralNodes,
      suspiciousPatterns,
      timestamp: Date.now()
    };
  }

  private addNode(nodeId: string, attributes: any): void {
    if (!this.adjacencyList.has(nodeId)) {
      this.adjacencyList.set(nodeId, new Set());
    }
    this.nodeAttributes.set(nodeId, attributes);
  }

  private addEdge(from: string, to: string, weight: number): void {
    this.adjacencyList.get(from)?.add(to);
    this.adjacencyList.get(to)?.add(from);
    
    const edgeKey = this.getEdgeKey(from, to);
    const currentWeight = this.edgeWeights.get(edgeKey) || 0;
    this.edgeWeights.set(edgeKey, currentWeight + weight);
  }

  private getEdgeKey(node1: string, node2: string): string {
    return node1 < node2 ? `${node1}-${node2}` : `${node2}-${node1}`;
  }

  private getEdges(): GraphEdge[] {
    const edges: GraphEdge[] = [];
    
    for (const [edgeKey, weight] of this.edgeWeights.entries()) {
      const [source, target] = edgeKey.split('-');
      edges.push({ source, target, weight });
    }
    
    return edges;
  }


  async detectCommunities(): Promise<CommunityDetectionResult[]> {
    const communities: Map<number, Set<string>> = new Map();
    const nodeToComm unity: Map<string, number> = new Map();
    
    let communityId = 0;
    for (const nodeId of this.adjacencyList.keys()) {
      if (!nodeToCommunity.has(nodeId)) {
        const community = this.expandCommunity(nodeId, nodeToCommunity, communityId);
        communities.set(communityId, community);
        communityId++;
      }
    }

    const results: CommunityDetectionResult[] = [];
    for (const [id, members] of communities.entries()) {
      const memberArray = Array.from(members);
      const density = this.calculateCommunityDensity(memberArray);
      const avgRisk = this.calculateAverageRisk(memberArray);
      
      results.push({
        communityId: id,
        members: memberArray,
        size: memberArray.length,
        density,
        averageRiskScore: avgRisk,
        isSuspicious: avgRisk > 0.7 || density > 0.8
      });
    }

    return results;
  }

  private expandCommunity(
    startNode: string,
    nodeToCommunity: Map<string, number>,
    communityId: number
  ): Set<string> {
    const community = new Set<string>();
    const queue = [startNode];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const node = queue.shift()!;
      
      if (visited.has(node)) continue;
      visited.add(node);
      
      community.add(node);
      nodeToCommunity.set(node, communityId);

      const neighbors = this.adjacencyList.get(node) || new Set();
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor) && !nodeToCommunity.has(neighbor)) {
          const edgeWeight = this.edgeWeights.get(this.getEdgeKey(node, neighbor)) || 0;
          if (edgeWeight > 100) {
            queue.push(neighbor);
          }
        }
      }
    }

    return community;
  }

  private calculateCommunityDensity(members: string[]): number {
    if (members.length < 2) return 0;

    let edgeCount = 0;
    const maxEdges = (members.length * (members.length - 1)) / 2;

    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const edgeKey = this.getEdgeKey(members[i], members[j]);
        if (this.edgeWeights.has(edgeKey)) {
          edgeCount++;
        }
      }
    }

    return edgeCount / maxEdges;
  }

  private calculateAverageRisk(members: string[]): number {
    let totalRisk = 0;
    
    for (const member of members) {
      const attrs = this.nodeAttributes.get(member);
      totalRisk += attrs?.riskScore || 0;
    }

    return totalRisk / members.length;
  }

  calculateCentrality(): Map<string, number> {
    const centrality = new Map<string, number>();
    
    for (const nodeId of this.adjacencyList.keys()) {
      const betweenness = this.calculateBetweennessCentrality(nodeId);
      const degree = this.calculateDegreeCentrality(nodeId);
      const closeness = this.calculateClosenessCentrality(nodeId);
      
      const combinedScore = (betweenness * 0.4) + (degree * 0.3) + (closeness * 0.3);
      centrality.set(nodeId, combinedScore);
    }

    return centrality;
  }

  private calculateBetweennessCentrality(nodeId: string): number {
    let betweenness = 0;
    const nodes = Array.from(this.adjacencyList.keys());

    for (const source of nodes) {
      if (source === nodeId) continue;
      
      for (const target of nodes) {
        if (target === nodeId || target === source) continue;
        
        const paths = this.findAllShortestPaths(source, target);
        const pathsThroughNode = paths.filter(path => path.includes(nodeId));
        
        if (paths.length > 0) {
          betweenness += pathsThroughNode.length / paths.length;
        }
      }
    }

    const n = nodes.length;
    const normalization = (n - 1) * (n - 2) / 2;
    return normalization > 0 ? betweenness / normalization : 0;
  }

  private calculateDegreeCentrality(nodeId: string): number {
    const degree = this.adjacencyList.get(nodeId)?.size || 0;
    const maxDegree = this.adjacencyList.size - 1;
    return maxDegree > 0 ? degree / maxDegree : 0;
  }

  private calculateClosenessCentrality(nodeId: string): number {
    const distances = this.dijkstra(nodeId);
    let totalDistance = 0;
    let reachableNodes = 0;

    for (const [targetId, distance] of distances.entries()) {
      if (targetId !== nodeId && distance < Infinity) {
        totalDistance += distance;
        reachableNodes++;
      }
    }

    if (reachableNodes === 0) return 0;
    
    const avgDistance = totalDistance / reachableNodes;
    return 1 / avgDistance;
  }

  private dijkstra(startNode: string): Map<string, number> {
    const distances = new Map<string, number>();
    const visited = new Set<string>();
    const queue: { node: string; distance: number }[] = [];

    for (const nodeId of this.adjacencyList.keys()) {
      distances.set(nodeId, Infinity);
    }
    distances.set(startNode, 0);
    queue.push({ node: startNode, distance: 0 });

    while (queue.length > 0) {
      queue.sort((a, b) => a.distance - b.distance);
      const { node: currentNode, distance: currentDistance } = queue.shift()!;

      if (visited.has(currentNode)) continue;
      visited.add(currentNode);

      const neighbors = this.adjacencyList.get(currentNode) || new Set();
      for (const neighbor of neighbors) {
        const edgeWeight = this.edgeWeights.get(this.getEdgeKey(currentNode, neighbor)) || 1;
        const newDistance = currentDistance + (1 / edgeWeight);

        if (newDistance < (distances.get(neighbor) || Infinity)) {
          distances.set(neighbor, newDistance);
          queue.push({ node: neighbor, distance: newDistance });
        }
      }
    }

    return distances;
  }

  private findAllShortestPaths(source: string, target: string): string[][] {
    const distances = new Map<string, number>();
    const predecessors = new Map<string, Set<string>>();
    const queue: { node: string; distance: number }[] = [];

    for (const nodeId of this.adjacencyList.keys()) {
      distances.set(nodeId, Infinity);
      predecessors.set(nodeId, new Set());
    }
    distances.set(source, 0);
    queue.push({ node: source, distance: 0 });

    while (queue.length > 0) {
      queue.sort((a, b) => a.distance - b.distance);
      const { node: currentNode, distance: currentDistance } = queue.shift()!;

      if (currentDistance > (distances.get(currentNode) || Infinity)) continue;

      const neighbors = this.adjacencyList.get(currentNode) || new Set();
      for (const neighbor of neighbors) {
        const edgeWeight = 1;
        const newDistance = currentDistance + edgeWeight;
        const neighborDistance = distances.get(neighbor) || Infinity;

        if (newDistance < neighborDistance) {
          distances.set(neighbor, newDistance);
          predecessors.set(neighbor, new Set([currentNode]));
          queue.push({ node: neighbor, distance: newDistance });
        } else if (newDistance === neighborDistance) {
          predecessors.get(neighbor)?.add(currentNode);
        }
      }
    }

    return this.reconstructPaths(source, target, predecessors);
  }

  private reconstructPaths(
    source: string,
    target: string,
    predecessors: Map<string, Set<string>>
  ): string[][] {
    if (source === target) return [[source]];

    const paths: string[][] = [];
    const preds = predecessors.get(target);
    
    if (!preds || preds.size === 0) return [];

    for (const pred of preds) {
      const subPaths = this.reconstructPaths(source, pred, predecessors);
      for (const subPath of subPaths) {
        paths.push([...subPath, target]);
      }
    }

    return paths;
  }

  async detectSuspiciousPatterns(): Promise<any[]> {
    const patterns = [];

    patterns.push(...this.detectCircularTransactions());
    patterns.push(...this.detectRapidFireTransactions());
    patterns.push(...this.detectMuleAccounts());
    patterns.push(...this.detectLayering());

    return patterns;
  }

  private detectCircularTransactions(): any[] {
    const circles = [];
    const visited = new Set<string>();

    for (const startNode of this.adjacencyList.keys()) {
      if (visited.has(startNode)) continue;

      const cycles = this.findCycles(startNode, visited);
      for (const cycle of cycles) {
        if (cycle.length >= 3 && cycle.length <= 6) {
          const totalAmount = this.calculateCycleAmount(cycle);
          circles.push({
            type: 'circular_transaction',
            nodes: cycle,
            amount: totalAmount,
            riskScore: 0.85,
            description: `Circular transaction detected involving ${cycle.length} accounts`
          });
        }
      }
    }

    return circles;
  }

  private findCycles(startNode: string, globalVisited: Set<string>): string[][] {
    const cycles: string[][] = [];
    const stack: { node: string; path: string[] }[] = [{ node: startNode, path: [startNode] }];
    const visited = new Set<string>();

    while (stack.length > 0) {
      const { node, path } = stack.pop()!;

      if (path.length > 6) continue;

      const neighbors = this.adjacencyList.get(node) || new Set();
      for (const neighbor of neighbors) {
        if (neighbor === startNode && path.length >= 3) {
          cycles.push([...path]);
        } else if (!path.includes(neighbor) && path.length < 6) {
          stack.push({ node: neighbor, path: [...path, neighbor] });
        }
      }

      visited.add(node);
    }

    globalVisited.add(startNode);
    return cycles;
  }

  private calculateCycleAmount(cycle: string[]): number {
    let totalAmount = 0;

    for (let i = 0; i < cycle.length; i++) {
      const from = cycle[i];
      const to = cycle[(i + 1) % cycle.length];
      const edgeKey = this.getEdgeKey(from, to);
      totalAmount += this.edgeWeights.get(edgeKey) || 0;
    }

    return totalAmount;
  }

  private detectRapidFireTransactions(): any[] {
    const rapidFire = [];
    
    for (const nodeId of this.adjacencyList.keys()) {
      const neighbors = this.adjacencyList.get(nodeId) || new Set();
      
      if (neighbors.size > 10) {
        let totalAmount = 0;
        for (const neighbor of neighbors) {
          const edgeKey = this.getEdgeKey(nodeId, neighbor);
          totalAmount += this.edgeWeights.get(edgeKey) || 0;
        }

        rapidFire.push({
          type: 'rapid_fire',
          node: nodeId,
          transactionCount: neighbors.size,
          totalAmount,
          riskScore: Math.min(0.95, 0.5 + (neighbors.size / 50)),
          description: `Account ${nodeId} made ${neighbors.size} transactions in short period`
        });
      }
    }

    return rapidFire;
  }

  private detectMuleAccounts(): any[] {
    const mules = [];

    for (const nodeId of this.adjacencyList.keys()) {
      const inDegree = this.calculateInDegree(nodeId);
      const outDegree = this.calculateOutDegree(nodeId);
      
      if (inDegree > 5 && outDegree > 5) {
        const inAmount = this.calculateInAmount(nodeId);
        const outAmount = this.calculateOutAmount(nodeId);
        const flowRatio = Math.abs(inAmount - outAmount) / Math.max(inAmount, outAmount);

        if (flowRatio < 0.2) {
          mules.push({
            type: 'mule_account',
            node: nodeId,
            inDegree,
            outDegree,
            inAmount,
            outAmount,
            riskScore: 0.90,
            description: `Potential mule account with balanced in/out flow`
          });
        }
      }
    }

    return mules;
  }

  private calculateInDegree(nodeId: string): number {
    let inDegree = 0;
    
    for (const [from, neighbors] of this.adjacencyList.entries()) {
      if (neighbors.has(nodeId) && from !== nodeId) {
        inDegree++;
      }
    }

    return inDegree;
  }

  private calculateOutDegree(nodeId: string): number {
    return this.adjacencyList.get(nodeId)?.size || 0;
  }

  private calculateInAmount(nodeId: string): number {
    let total = 0;
    
    for (const [from, neighbors] of this.adjacencyList.entries()) {
      if (neighbors.has(nodeId) && from !== nodeId) {
        const edgeKey = this.getEdgeKey(from, nodeId);
        total += this.edgeWeights.get(edgeKey) || 0;
      }
    }

    return total;
  }

  private calculateOutAmount(nodeId: string): number {
    let total = 0;
    const neighbors = this.adjacencyList.get(nodeId) || new Set();
    
    for (const neighbor of neighbors) {
      const edgeKey = this.getEdgeKey(nodeId, neighbor);
      total += this.edgeWeights.get(edgeKey) || 0;
    }

    return total;
  }

  private detectLayering(): any[] {
    const layering = [];

    for (const nodeId of this.adjacencyList.keys()) {
      const paths = this.findLongPaths(nodeId, 4);
      
      for (const path of paths) {
        const totalAmount = this.calculatePathAmount(path);
        const avgHopAmount = totalAmount / path.length;

        if (avgHopAmount > 1000 && path.length >= 4) {
          layering.push({
            type: 'layering',
            path,
            length: path.length,
            totalAmount,
            riskScore: 0.80,
            description: `Layering pattern detected through ${path.length} intermediaries`
          });
        }
      }
    }

    return layering;
  }

  private findLongPaths(startNode: string, minLength: number): string[][] {
    const paths: string[][] = [];
    const stack: { node: string; path: string[] }[] = [{ node: startNode, path: [startNode] }];

    while (stack.length > 0) {
      const { node, path } = stack.pop()!;

      if (path.length >= minLength) {
        paths.push([...path]);
      }

      if (path.length < 8) {
        const neighbors = this.adjacencyList.get(node) || new Set();
        for (const neighbor of neighbors) {
          if (!path.includes(neighbor)) {
            stack.push({ node: neighbor, path: [...path, neighbor] });
          }
        }
      }
    }

    return paths.filter(p => p.length >= minLength);
  }

  private calculatePathAmount(path: string[]): number {
    let total = 0;

    for (let i = 0; i < path.length - 1; i++) {
      const edgeKey = this.getEdgeKey(path[i], path[i + 1]);
      total += this.edgeWeights.get(edgeKey) || 0;
    }

    return total;
  }

  async calculatePageRank(iterations: number = 20, dampingFactor: number = 0.85): Promise<Map<string, number>> {
    const nodes = Array.from(this.adjacencyList.keys());
    const n = nodes.length;
    const pageRank = new Map<string, number>();
    
    for (const node of nodes) {
      pageRank.set(node, 1 / n);
    }

    for (let iter = 0; iter < iterations; iter++) {
      const newPageRank = new Map<string, number>();

      for (const node of nodes) {
        let rank = (1 - dampingFactor) / n;

        for (const [source, neighbors] of this.adjacencyList.entries()) {
          if (neighbors.has(node)) {
            const sourceRank = pageRank.get(source) || 0;
            const outDegree = neighbors.size;
            rank += dampingFactor * (sourceRank / outDegree);
          }
        }

        newPageRank.set(node, rank);
      }

      for (const [node, rank] of newPageRank.entries()) {
        pageRank.set(node, rank);
      }
    }

    return pageRank;
  }
}

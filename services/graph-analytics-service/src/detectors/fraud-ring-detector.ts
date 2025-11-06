/**
 * Fraud Ring Detector
 * Identifies organized fraud rings using graph patterns
 */

interface GraphData {
  nodes: Array<{ id: string; type: string; properties: Record<string, any> }>;
  edges: Array<{ source: string; target: string; type: string; weight: number; properties: Record<string, any> }>;
}

export class FraudRingDetector {
  detectPatterns(graph: GraphData): string[] {
    const patterns: string[] = [];
    
    // Pattern 1: Shared device usage
    if (this.detectSharedDevicePattern(graph)) {
      patterns.push('SHARED_DEVICE');
    }
    
    // Pattern 2: Circular money flow
    if (this.detectCircularFlow(graph)) {
      patterns.push('CIRCULAR_FLOW');
    }
    
    // Pattern 3: Rapid account creation
    if (this.detectRapidAccountCreation(graph)) {
      patterns.push('RAPID_ACCOUNT_CREATION');
    }
    
    // Pattern 4: Shared IP addresses
    if (this.detectSharedIPPattern(graph)) {
      patterns.push('SHARED_IP');
    }
    
    // Pattern 5: Coordinated transactions
    if (this.detectCoordinatedTransactions(graph)) {
      patterns.push('COORDINATED_TRANSACTIONS');
    }
    
    return patterns;
  }
  
  private detectSharedDevicePattern(graph: GraphData): boolean {
    const deviceNodes = graph.nodes.filter(n => n.type === 'device');
    const userConnections = new Map<string, Set<string>>();
    
    graph.edges.forEach(edge => {
      if (edge.type === 'USES_DEVICE') {
        if (!userConnections.has(edge.target)) {
          userConnections.set(edge.target, new Set());
        }
        userConnections.get(edge.target)?.add(edge.source);
      }
    });
    
    // Check if any device is shared by multiple users
    for (const [device, users] of userConnections.entries()) {
      if (users.size >= 3) {
        return true;
      }
    }
    
    return false;
  }
  
  private detectCircularFlow(graph: GraphData): boolean {
    // Detect cycles in transaction graph
    const visited = new Set<string>();
    const recStack = new Set<string>();
    
    const hasCycle = (nodeId: string): boolean => {
      visited.add(nodeId);
      recStack.add(nodeId);
      
      const outgoingEdges = graph.edges.filter(e => e.source === nodeId && e.type === 'TRANSACTION');
      
      for (const edge of outgoingEdges) {
        if (!visited.has(edge.target)) {
          if (hasCycle(edge.target)) {
            return true;
          }
        } else if (recStack.has(edge.target)) {
          return true;
        }
      }
      
      recStack.delete(nodeId);
      return false;
    };
    
    for (const node of graph.nodes) {
      if (node.type === 'user' && !visited.has(node.id)) {
        if (hasCycle(node.id)) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  private detectRapidAccountCreation(graph: GraphData): boolean {
    const userNodes = graph.nodes.filter(n => n.type === 'user');
    const creationTimes = userNodes
      .map(n => n.properties.createdAt)
      .filter(t => t !== undefined)
      .sort((a, b) => a - b);
    
    // Check if multiple accounts created within short time window
    for (let i = 0; i < creationTimes.length - 2; i++) {
      const timeWindow = creationTimes[i + 2] - creationTimes[i];
      if (timeWindow < 3600000) { // 1 hour
        return true;
      }
    }
    
    return false;
  }
  
  private detectSharedIPPattern(graph: GraphData): boolean {
    const ipConnections = new Map<string, Set<string>>();
    
    graph.edges.forEach(edge => {
      if (edge.type === 'USES_IP') {
        if (!ipConnections.has(edge.target)) {
          ipConnections.set(edge.target, new Set());
        }
        ipConnections.get(edge.target)?.add(edge.source);
      }
    });
    
    for (const [ip, users] of ipConnections.entries()) {
      if (users.size >= 5) {
        return true;
      }
    }
    
    return false;
  }
  
  private detectCoordinatedTransactions(graph: GraphData): boolean {
    const transactions = graph.edges.filter(e => e.type === 'TRANSACTION');
    
    // Group transactions by time window
    const timeWindows = new Map<number, Array<typeof transactions[0]>>();
    const windowSize = 300000; // 5 minutes
    
    transactions.forEach(tx => {
      const timestamp = tx.properties.timestamp || 0;
      const window = Math.floor(timestamp / windowSize);
      
      if (!timeWindows.has(window)) {
        timeWindows.set(window, []);
      }
      timeWindows.get(window)?.push(tx);
    });
    
    // Check for coordinated activity
    for (const [window, txs] of timeWindows.entries()) {
      if (txs.length >= 5) {
        // Check if transactions involve related users
        const users = new Set(txs.flatMap(tx => [tx.source, tx.target]));
        if (users.size <= txs.length / 2) {
          return true;
        }
      }
    }
    
    return false;
  }
}

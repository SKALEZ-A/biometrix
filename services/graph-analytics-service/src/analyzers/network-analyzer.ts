/**
 * Network Analyzer
 * Analyzes network structure and detects anomalies
 */

interface GraphData {
  nodes: Array<{ id: string; type: string; properties: Record<string, any> }>;
  edges: Array<{ source: string; target: string; type: string; weight: number }>;
}

interface NetworkMetrics {
  degree: number;
  pageRank: number;
  clusteringCoefficient: number;
}

export class NetworkAnalyzer {
  analyzeStructure(graph: GraphData): any {
    return {
      nodeCount: graph.nodes.length,
      edgeCount: graph.edges.length,
      density: this.calculateDensity(graph),
      avgDegree: this.calculateAverageDegree(graph),
      components: this.findConnectedComponents(graph)
    };
  }
  
  detectAnomalies(metrics: NetworkMetrics, structure: any): string[] {
    const anomalies: string[] = [];
    
    if (metrics.degree > 100) {
      anomalies.push('UNUSUALLY_HIGH_DEGREE');
    }
    
    if (metrics.clusteringCoefficient < 0.1 && metrics.degree > 10) {
      anomalies.push('LOW_CLUSTERING_HIGH_DEGREE');
    }
    
    if (metrics.pageRank > 0.01) {
      anomalies.push('HIGH_INFLUENCE');
    }
    
    return anomalies;
  }
  
  private calculateDensity(graph: GraphData): number {
    const n = graph.nodes.length;
    if (n < 2) return 0;
    return (2 * graph.edges.length) / (n * (n - 1));
  }
  
  private calculateAverageDegree(graph: GraphData): number {
    const degrees = new Map<string, number>();
    
    graph.nodes.forEach(node => degrees.set(node.id, 0));
    
    graph.edges.forEach(edge => {
      degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
      degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
    });
    
    const total = Array.from(degrees.values()).reduce((sum, d) => sum + d, 0);
    return total / graph.nodes.length;
  }
  
  private findConnectedComponents(graph: GraphData): number {
    const visited = new Set<string>();
    let components = 0;
    
    const dfs = (nodeId: string) => {
      visited.add(nodeId);
      const neighbors = graph.edges
        .filter(e => e.source === nodeId || e.target === nodeId)
        .map(e => e.source === nodeId ? e.target : e.source);
      
      neighbors.forEach(neighbor => {
        if (!visited.has(neighbor)) {
          dfs(neighbor);
        }
      });
    };
    
    graph.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id);
        components++;
      }
    });
    
    return components;
  }
}

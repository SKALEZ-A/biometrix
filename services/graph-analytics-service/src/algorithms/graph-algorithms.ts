/**
 * Graph Algorithms Implementation
 * Advanced graph algorithms for fraud detection
 */

export interface GraphNode {
  id: string;
  properties: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export class GraphAlgorithms {
  /**
   * Calculate PageRank for all nodes
   */
  static pageRank(graph: GraphData, dampingFactor: number = 0.85, maxIterations: number = 100, tolerance: number = 1e-6): Map<string, number> {
    const nodeCount = graph.nodes.length;
    const scores = new Map<string, number>();
    const newScores = new Map<string, number>();
    
    // Initialize scores
    graph.nodes.forEach(node => {
      scores.set(node.id, 1 / nodeCount);
    });
    
    // Build adjacency list
    const outLinks = new Map<string, string[]>();
    const inLinks = new Map<string, string[]>();
    
    graph.nodes.forEach(node => {
      outLinks.set(node.id, []);
      inLinks.set(node.id, []);
    });
    
    graph.edges.forEach(edge => {
      outLinks.get(edge.source)?.push(edge.target);
      inLinks.get(edge.target)?.push(edge.source);
    });
    
    // Iterate
    for (let iter = 0; iter < maxIterations; iter++) {
      let diff = 0;
      
      graph.nodes.forEach(node => {
        let sum = 0;
        const incoming = inLinks.get(node.id) || [];
        
        incoming.forEach(sourceId => {
          const sourceScore = scores.get(sourceId) || 0;
          const sourceOutDegree = outLinks.get(sourceId)?.length || 1;
          sum += sourceScore / sourceOutDegree;
        });
        
        const newScore = (1 - dampingFactor) / nodeCount + dampingFactor * sum;
        newScores.set(node.id, newScore);
        
        diff += Math.abs(newScore - (scores.get(node.id) || 0));
      });
      
      // Update scores
      newScores.forEach((score, nodeId) => {
        scores.set(nodeId, score);
      });
      
      if (diff < tolerance) {
        break;
      }
    }
    
    return scores;
  }
  
  /**
   * Detect communities using Louvain algorithm
   */
  static louvainCommunityDetection(graph: GraphData): Map<string, number> {
    const communities = new Map<string, number>();
    
    // Initialize each node in its own community
    graph.nodes.forEach((node, index) => {
      communities.set(node.id, index);
    });
    
    // Build weighted adjacency matrix
    const weights = new Map<string, Map<string, number>>();
    graph.nodes.forEach(node => {
      weights.set(node.id, new Map());
    });
    
    graph.edges.forEach(edge => {
      weights.get(edge.source)?.set(edge.target, edge.weight);
      weights.get(edge.target)?.set(edge.source, edge.weight);
    });
    
    let improved = true;
    let iteration = 0;
    const maxIterations = 100;
    
    while (improved && iteration < maxIterations) {
      improved = false;
      iteration++;
      
      // For each node, try moving to neighbor communities
      for (const node of graph.nodes) {
        const currentCommunity = communities.get(node.id)!;
        let bestCommunity = currentCommunity;
        let bestGain = 0;
        
        // Get neighbor communities
        const neighborCommunities = new Set<number>();
        const nodeWeights = weights.get(node.id) || new Map();
        
        nodeWeights.forEach((weight, neighborId) => {
          const neighborCommunity = communities.get(neighborId);
          if (neighborCommunity !== undefined) {
            neighborCommunities.add(neighborCommunity);
          }
        });
        
        // Try each neighbor community
        neighborCommunities.forEach(community => {
          if (community === currentCommunity) return;
          
          const gain = this.modularityGain(node.id, community, communities, weights);
          
          if (gain > bestGain) {
            bestGain = gain;
            bestCommunity = community;
          }
        });
        
        if (bestCommunity !== currentCommunity) {
          communities.set(node.id, bestCommunity);
          improved = true;
        }
      }
    }
    
    return communities;
  }
  
  private static modularityGain(
    nodeId: string,
    targetCommunity: number,
    communities: Map<string, number>,
    weights: Map<string, Map<string, number>>
  ): number {
    // Simplified modularity gain calculation
    let gain = 0;
    const nodeWeights = weights.get(nodeId) || new Map();
    
    nodeWeights.forEach((weight, neighborId) => {
      const neighborCommunity = communities.get(neighborId);
      if (neighborCommunity === targetCommunity) {
        gain += weight;
      }
    });
    
    return gain;
  }
  
  /**
   * Find strongly connected components
   */
  static stronglyConnectedComponents(graph: GraphData): string[][] {
    const visited = new Set<string>();
    const stack: string[] = [];
    const components: string[][] = [];
    
    // Build adjacency lists
    const adjList = new Map<string, string[]>();
    const reverseAdjList = new Map<string, string[]>();
    
    graph.nodes.forEach(node => {
      adjList.set(node.id, []);
      reverseAdjList.set(node.id, []);
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.push(edge.target);
      reverseAdjList.get(edge.target)?.push(edge.source);
    });
    
    // First DFS to fill stack
    const dfs1 = (nodeId: string) => {
      visited.add(nodeId);
      const neighbors = adjList.get(nodeId) || [];
      
      neighbors.forEach(neighbor => {
        if (!visited.has(neighbor)) {
          dfs1(neighbor);
        }
      });
      
      stack.push(nodeId);
    };
    
    graph.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs1(node.id);
      }
    });
    
    // Second DFS on reversed graph
    visited.clear();
    
    const dfs2 = (nodeId: string, component: string[]) => {
      visited.add(nodeId);
      component.push(nodeId);
      
      const neighbors = reverseAdjList.get(nodeId) || [];
      neighbors.forEach(neighbor => {
        if (!visited.has(neighbor)) {
          dfs2(neighbor, component);
        }
      });
    };
    
    while (stack.length > 0) {
      const nodeId = stack.pop()!;
      if (!visited.has(nodeId)) {
        const component: string[] = [];
        dfs2(nodeId, component);
        components.push(component);
      }
    }
    
    return components;
  }
  
  /**
   * Calculate betweenness centrality
   */
  static betweennessCentrality(graph: GraphData): Map<string, number> {
    const centrality = new Map<string, number>();
    
    // Initialize
    graph.nodes.forEach(node => {
      centrality.set(node.id, 0);
    });
    
    // Build adjacency list
    const adjList = new Map<string, string[]>();
    graph.nodes.forEach(node => {
      adjList.set(node.id, []);
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.push(edge.target);
      adjList.get(edge.target)?.push(edge.source);
    });
    
    // For each node as source
    graph.nodes.forEach(source => {
      const stack: string[] = [];
      const paths = new Map<string, string[][]>();
      const sigma = new Map<string, number>();
      const distance = new Map<string, number>();
      const delta = new Map<string, number>();
      
      // Initialize
      graph.nodes.forEach(node => {
        paths.set(node.id, []);
        sigma.set(node.id, 0);
        distance.set(node.id, -1);
        delta.set(node.id, 0);
      });
      
      sigma.set(source.id, 1);
      distance.set(source.id, 0);
      
      // BFS
      const queue: string[] = [source.id];
      
      while (queue.length > 0) {
        const v = queue.shift()!;
        stack.push(v);
        
        const neighbors = adjList.get(v) || [];
        neighbors.forEach(w => {
          // First time visiting w?
          if (distance.get(w) === -1) {
            queue.push(w);
            distance.set(w, (distance.get(v) || 0) + 1);
          }
          
          // Shortest path to w via v?
          if (distance.get(w) === (distance.get(v) || 0) + 1) {
            sigma.set(w, (sigma.get(w) || 0) + (sigma.get(v) || 0));
            paths.get(w)?.push([v]);
          }
        });
      }
      
      // Accumulation
      while (stack.length > 0) {
        const w = stack.pop()!;
        const wPaths = paths.get(w) || [];
        
        wPaths.forEach(path => {
          const v = path[0];
          const c = ((sigma.get(v) || 0) / (sigma.get(w) || 1)) * (1 + (delta.get(w) || 0));
          delta.set(v, (delta.get(v) || 0) + c);
        });
        
        if (w !== source.id) {
          centrality.set(w, (centrality.get(w) || 0) + (delta.get(w) || 0));
        }
      }
    });
    
    // Normalize
    const n = graph.nodes.length;
    const normFactor = ((n - 1) * (n - 2)) / 2;
    
    centrality.forEach((value, nodeId) => {
      centrality.set(nodeId, value / normFactor);
    });
    
    return centrality;
  }
  
  /**
   * Calculate clustering coefficient
   */
  static clusteringCoefficient(graph: GraphData): Map<string, number> {
    const coefficients = new Map<string, number>();
    
    // Build adjacency list
    const adjList = new Map<string, Set<string>>();
    graph.nodes.forEach(node => {
      adjList.set(node.id, new Set());
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.add(edge.target);
      adjList.get(edge.target)?.add(edge.source);
    });
    
    // Calculate for each node
    graph.nodes.forEach(node => {
      const neighbors = Array.from(adjList.get(node.id) || []);
      const k = neighbors.length;
      
      if (k < 2) {
        coefficients.set(node.id, 0);
        return;
      }
      
      // Count triangles
      let triangles = 0;
      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          if (adjList.get(neighbors[i])?.has(neighbors[j])) {
            triangles++;
          }
        }
      }
      
      const possibleTriangles = (k * (k - 1)) / 2;
      coefficients.set(node.id, triangles / possibleTriangles);
    });
    
    return coefficients;
  }
  
  /**
   * Find k-core decomposition
   */
  static kCoreDecomposition(graph: GraphData): Map<string, number> {
    const coreness = new Map<string, number>();
    const degrees = new Map<string, number>();
    
    // Build adjacency list and calculate degrees
    const adjList = new Map<string, Set<string>>();
    graph.nodes.forEach(node => {
      adjList.set(node.id, new Set());
      degrees.set(node.id, 0);
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.add(edge.target);
      adjList.get(edge.target)?.add(edge.source);
    });
    
    adjList.forEach((neighbors, nodeId) => {
      degrees.set(nodeId, neighbors.size);
    });
    
    // Sort nodes by degree
    const sortedNodes = Array.from(graph.nodes).sort((a, b) => {
      return (degrees.get(a.id) || 0) - (degrees.get(b.id) || 0);
    });
    
    const removed = new Set<string>();
    
    sortedNodes.forEach(node => {
      const degree = degrees.get(node.id) || 0;
      coreness.set(node.id, degree);
      removed.add(node.id);
      
      // Update neighbors' degrees
      const neighbors = adjList.get(node.id) || new Set();
      neighbors.forEach(neighbor => {
        if (!removed.has(neighbor)) {
          degrees.set(neighbor, (degrees.get(neighbor) || 0) - 1);
        }
      });
    });
    
    return coreness;
  }
  
  /**
   * Detect cycles in graph
   */
  static detectCycles(graph: GraphData): string[][] {
    const cycles: string[][] = [];
    const visited = new Set<string>();
    const recStack = new Set<string>();
    const path: string[] = [];
    
    // Build adjacency list
    const adjList = new Map<string, string[]>();
    graph.nodes.forEach(node => {
      adjList.set(node.id, []);
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.push(edge.target);
    });
    
    const dfs = (nodeId: string): boolean => {
      visited.add(nodeId);
      recStack.add(nodeId);
      path.push(nodeId);
      
      const neighbors = adjList.get(nodeId) || [];
      
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (dfs(neighbor)) {
            return true;
          }
        } else if (recStack.has(neighbor)) {
          // Found cycle
          const cycleStart = path.indexOf(neighbor);
          const cycle = path.slice(cycleStart);
          cycles.push(cycle);
        }
      }
      
      path.pop();
      recStack.delete(nodeId);
      return false;
    };
    
    graph.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id);
      }
    });
    
    return cycles;
  }
  
  /**
   * Calculate graph density
   */
  static graphDensity(graph: GraphData): number {
    const n = graph.nodes.length;
    const m = graph.edges.length;
    
    if (n < 2) return 0;
    
    const maxEdges = (n * (n - 1)) / 2;
    return m / maxEdges;
  }
  
  /**
   * Find maximum clique
   */
  static findMaximumClique(graph: GraphData): string[] {
    // Build adjacency list
    const adjList = new Map<string, Set<string>>();
    graph.nodes.forEach(node => {
      adjList.set(node.id, new Set());
    });
    
    graph.edges.forEach(edge => {
      adjList.get(edge.source)?.add(edge.target);
      adjList.get(edge.target)?.add(edge.source);
    });
    
    let maxClique: string[] = [];
    
    const bronKerbosch = (r: Set<string>, p: Set<string>, x: Set<string>) => {
      if (p.size === 0 && x.size === 0) {
        if (r.size > maxClique.length) {
          maxClique = Array.from(r);
        }
        return;
      }
      
      const pArray = Array.from(p);
      
      for (const v of pArray) {
        const neighbors = adjList.get(v) || new Set();
        
        const newR = new Set(r);
        newR.add(v);
        
        const newP = new Set<string>();
        p.forEach(node => {
          if (neighbors.has(node)) newP.add(node);
        });
        
        const newX = new Set<string>();
        x.forEach(node => {
          if (neighbors.has(node)) newX.add(node);
        });
        
        bronKerbosch(newR, newP, newX);
        
        p.delete(v);
        x.add(v);
      }
    };
    
    const allNodes = new Set(graph.nodes.map(n => n.id));
    bronKerbosch(new Set(), allNodes, new Set());
    
    return maxClique;
  }
}

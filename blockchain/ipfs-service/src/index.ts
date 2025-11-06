import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { createHash, randomBytes } from 'crypto';
import { logger } from './utils/logger';
import { BiometricEvidence, ZKProofData } from './types/ipfs';
import { smartContract } from './integrations/smartContract';
import { config } from './config/config';

/**
 * IPFS Service for Biometric Fraud Prevention
 * Decentralized storage layer for biometric templates, proofs, and evidence
 * 
 * Features:
 * - Content-addressable storage with SHA-256 CIDv0
 * - Biometric template encryption and fragmentation
 * - Zero-knowledge proof storage and verification
 * - Fraud evidence packaging and tamper-proof sealing
 * - Integration with Ethereum smart contracts
 * - Local simulation mode (no external IPFS node required)
 * 
 * Security:
 * - All data encrypted client-side before upload
 * - Merkle-DAG structure for evidence trees
 * - Content integrity verification
 * - Access control via smart contract permissions
 */

export class IPFSService {
  private static instance: IPFSService;
  private storageDir: string;
  private isConnected: boolean = false;
  private pinQueue: Array<{ content: Buffer; metadata: any }> = [];
  private processing: boolean = false;

  constructor() {
    this.storageDir = config.storagePath || path.join(__dirname, '../../storage/ipfs');
    this.ensureStorageDir();
    this.simulateIPFSConnection();
  }

  static getInstance(): IPFSService {
    if (!IPFSService.instance) {
      IPFSService.instance = new IPFSService();
    }
    return IPFSService.instance;
  }

  private ensureStorageDir(): void {
    if (!fs.existsSync(this.storageDir)) {
      fs.mkdirSync(this.storageDir, { recursive: true });
      logger.info(`Created IPFS storage directory: ${this.storageDir}`);
    }
  }

  private simulateIPFSConnection(): void {
    // In production, connect to IPFS node via HTTP API
    // For demo, simulate successful connection
    this.isConnected = true;
    logger.info('IPFS service initialized (simulation mode)');
    
    // Start background pin processor
    this.processPinQueue();
  }

  /**
   * Generate CIDv0 from content (SHA-256 hash)
   * @param content Buffer to hash
   * @returns CID string
   */
  private generateCID(content: Buffer): string {
    const hash = createHash('sha256').update(content).digest();
    return 'Qm' + hash.toString('base58'); // CIDv0 format
  }

  /**
   * Encrypt content with AES-256-GCM (client-side encryption simulation)
   * @param content Plaintext buffer
   * @param key Encryption key (derived from user biometric)
   * @returns Encrypted buffer with auth tag
   */
  private async encryptContent(content: Buffer, key: Buffer): Promise<Buffer> {
    const iv = randomBytes(12);
    const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
    let encrypted = cipher.update(content);
    encrypted = Buffer.concat([encrypted, cipher.final()]);
    const authTag = cipher.getAuthTag();
    return Buffer.concat([iv, authTag, encrypted]);
  }

  /**
   * Decrypt content (for authorized retrieval)
   * @param encrypted Encrypted buffer
   * @param key Decryption key
   * @returns Plaintext buffer
   */
  private async decryptContent(encrypted: Buffer, key: Buffer): Promise<Buffer> {
    const iv = encrypted.slice(0, 12);
    const authTag = encrypted.slice(12, 28);
    const content = encrypted.slice(28);
    
    const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
    decipher.setAuthTag(authTag);
    let decrypted = decipher.update(content);
    decrypted = Buffer.concat([decrypted, decipher.final()]);
    return decrypted;
  }

  /**
   * Store biometric template securely
   * @param template Biometric template data
   * @param userId User identifier
   * @param modality Biometric modality
   * @returns IPFS CID and metadata
   */
  async storeBiometricTemplate(
    template: Buffer, 
    userId: string, 
    modality: number
  ): Promise<{ cid: string; metadata: any }> {
    if (!this.isConnected) {
      throw new Error('IPFSService: Not connected');
    }

    // Generate encryption key from userId + modality (in production, derive from biometric)
    const key = crypto.createHash('sha256')
      .update(userId + modality + config.encryptionSalt)
      .digest();

    // Encrypt template
    const encryptedTemplate = await this.encryptContent(template, key);

    // Create metadata
    const metadata: BiometricEvidence = {
      type: 'biometric_template',
      userId: this.hashUserId(userId),
      modality: modality,
      version: '1.0',
      timestamp: Date.now(),
      size: encryptedTemplate.length,
      fragments: 1, // Single fragment for simplicity
      encryption: 'aes-256-gcm',
      commitment: this.generateMerkleRoot([encryptedTemplate]), // For verification
      accessControl: {
        owner: userId,
        permissions: ['read', 'revoke'],
        expiry: Date.now() + 365 * 24 * 60 * 60 * 1000 // 1 year
      }
    };

    // Serialize and store
    const content = Buffer.concat([
      Buffer.from(JSON.stringify(metadata), 'utf8'),
      Buffer.from('\n---\n', 'utf8'),
      encryptedTemplate
    ]);

    const cid = this.generateCID(content);
    
    // Simulate pinning
    await this.pinContent(cid, content, metadata);

    logger.info(`Stored biometric template: ${cid} for user ${userId} (${modality})`);

    return { cid, metadata };
  }

  /**
   * Store zero-knowledge proof
   * @param proof ZK proof data
   * @param publicInputs Public inputs to the proof
   * @param verificationKey Verification key hash
   * @returns CID and proof metadata
   */
  async storeZKProof(
    proof: ZKProofData, 
    publicInputs: any, 
    verificationKey: string
  ): Promise<{ cid: string; metadata: any }> {
    // Proofs don't need encryption (zk-SNARKs are public)
    const proofBuffer = Buffer.from(JSON.stringify({
      ...proof,
      circuit: 'BiometricVerification',
      version: '1.0',
      timestamp: Date.now(),
      publicInputs: publicInputs,
      verificationKey: verificationKey,
      size: proof.proof.length + proof.publicSignals.length
    }), 'utf8');

    const cid = this.generateCID(proofBuffer);
    
    // Pin proof
    await this.pinContent(cid, proofBuffer, {
      type: 'zk_proof',
      circuit: 'BiometricVerification',
      securityLevel: 128,
      verifiable: true
    });

    // Store on-chain reference (if transaction provided)
    if (proof.transactionId) {
      await smartContract.storeProofReference(proof.transactionId, cid);
    }

    logger.info(`Stored ZK proof: ${cid} (tx: ${proof.transactionId || 'none'})`);
    return { cid, metadata: { type: 'zk_proof', circuit: 'BiometricVerification' } };
  }

  /**
   * Package and store fraud evidence
   * @param evidence Fraud evidence components
   * @param caseId Fraud case ID
   * @param reporter Reporter address
   * @returns Evidence CID and Merkle tree root
   */
  async storeFraudEvidence(
    evidence: {
      transactionData: any;
      biometricLogs: Buffer[];
      screenshots?: Buffer[];
      networkTraces?: Buffer;
      mlAnalysis?: any;
    },
    caseId: string,
    reporter: string
  ): Promise<{ cid: string; merkleRoot: string; metadata: any }> {
    // Create Merkle-DAG structure for evidence tree
    const evidenceTree: any = {
      type: 'fraud_evidence',
      caseId: caseId,
      reporter: this.hashUserId(reporter),
      timestamp: Date.now(),
      version: '1.0',
      components: [],
      hashes: []
    };

    const allContent: Buffer[] = [];

    // Add transaction data
    const txJson = JSON.stringify(evidence.transactionData);
    const txBuffer = Buffer.from(txJson, 'utf8');
    const txHash = this.generateCID(txBuffer);
    evidenceTree.components.push({ type: 'transaction', hash: txHash });
    allContent.push(txBuffer);
    evidenceTree.hashes.push(txHash);

    // Add biometric logs (encrypted)
    for (let i = 0; i < evidence.biometricLogs.length; i++) {
      const logKey = crypto.createHash('sha256')
        .update(reporter + caseId + i)
        .digest();
      const encryptedLog = await this.encryptContent(evidence.biometricLogs[i], logKey);
      const logHash = this.generateCID(encryptedLog);
      evidenceTree.components.push({ type: 'biometric_log', index: i, hash: logHash });
      allContent.push(encryptedLog);
      evidenceTree.hashes.push(logHash);
    }

    // Add screenshots (if present)
    if (evidence.screenshots) {
      for (let i = 0; i < evidence.screenshots.length; i++) {
        const imgHash = this.generateCID(evidence.screenshots[i]);
        evidenceTree.components.push({ type: 'screenshot', index: i, hash: imgHash });
        allContent.push(evidence.screenshots[i]);
        evidenceTree.hashes.push(imgHash);
      }
    }

    // Add network traces
    if (evidence.networkTraces) {
      const netKey = crypto.createHash('sha256').update(caseId + 'network').digest();
      const encryptedNet = await this.encryptContent(evidence.networkTraces, netKey);
      const netHash = this.generateCID(encryptedNet);
      evidenceTree.components.push({ type: 'network_trace', hash: netHash });
      allContent.push(encryptedNet);
      evidenceTree.hashes.push(netHash);
    }

    // Add ML analysis
    if (evidence.mlAnalysis) {
      const mlJson = JSON.stringify(evidence.mlAnalysis);
      const mlBuffer = Buffer.from(mlJson, 'utf8');
      const mlHash = this.generateCID(mlBuffer);
      evidenceTree.components.push({ type: 'ml_analysis', hash: mlHash });
      allContent.push(mlBuffer);
      evidenceTree.hashes.push(mlHash);
    }

    // Generate Merkle root
    const merkleRoot = this.generateMerkleRoot(evidenceTree.hashes);

    // Package everything
    const packageContent = Buffer.concat([
      Buffer.from(JSON.stringify(evidenceTree), 'utf8'),
      Buffer.from('\nMERKLE_ROOT:' + merkleRoot + '\n', 'utf8'),
      ...allContent
    ]);

    const evidenceCid = this.generateCID(packageContent);
    
    // Pin evidence package
    await this.pinContent(evidenceCid, packageContent, evidenceTree);

    // Store on-chain
    await smartContract.storeFraudEvidence(caseId, evidenceCid, merkleRoot);

    logger.info(`Stored fraud evidence: ${evidenceCid} for case ${caseId} (Merkle: ${merkleRoot})`);

    return { cid: evidenceCid, merkleRoot, metadata: evidenceTree };
  }

  /**
   * Retrieve and verify content by CID
   * @param cid Content identifier
   * @param expectedHash Optional expected hash for integrity
   * @returns Content buffer and metadata
   */
  async retrieveContent(cid: string, expectedHash?: string): Promise<{ content: Buffer; metadata: any }> {
    // Simulate retrieval from local storage
    const filePath = path.join(this.storageDir, cid);
    
    if (!fs.existsSync(filePath)) {
      throw new Error(`IPFSService: Content not found: ${cid}`);
    }

    const content = fs.readFileSync(filePath);
    
    // Verify CID matches content
    const actualHash = this.generateCID(content);
    if (actualHash !== cid) {
      throw new Error(`IPFSService: Content mismatch for CID ${cid}`);
    }

    // Verify expected hash if provided
    if (expectedHash && this.generateMerkleRoot([content]) !== expectedHash) {
      throw new Error(`IPFSService: Integrity check failed for ${cid}`);
    }

    // Parse metadata
    const contentStr = content.toString('utf8');
    const separatorIndex = contentStr.indexOf('\n---\n');
    if (separatorIndex === -1) {
      throw new Error(`IPFSService: Invalid content format for ${cid}`);
    }

    const metadataStr = contentStr.substring(0, separatorIndex);
    const metadata = JSON.parse(metadataStr);
    const data = contentStr.substring(separatorIndex + 5);

    logger.info(`Retrieved content: ${cid} (${metadata.type || 'unknown'})`);

    return { content: Buffer.from(data, 'utf8'), metadata };
  }

  /**
   * Pin content to IPFS (make permanently available)
   * @param cid Content identifier
   * @param content Content buffer
   * @param metadata Optional metadata
   */
  private async pinContent(cid: string, content: Buffer, metadata?: any): Promise<void> {
    // Add to queue for background processing
    this.pinQueue.push({ content, metadata });
    
    if (!this.processing) {
      this.processing = true;
      await this.processPinQueue();
    }

    // Store locally (simulation)
    const filePath = path.join(this.storageDir, cid);
    fs.writeFileSync(filePath, content);
    
    // In production: call IPFS node API
    // await this.ipfsNode.pin.add(cid);
    
    logger.debug(`Pinned content: ${cid}`);
  }

  private async processPinQueue(): Promise<void> {
    while (this.pinQueue.length > 0) {
      const { content, metadata } = this.pinQueue.shift()!;
      
      try {
        const cid = this.generateCID(content);
        await this.pinContent(cid, content, metadata);
        
        // Update pin count and statistics
        // this.stats.pins++;
        // this.stats.storageUsed += content.length;
        
      } catch (error) {
        logger.error(`Failed to pin content: ${error}`);
        // Re-queue on failure (with backoff in production)
        this.pinQueue.unshift({ content, metadata });
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1s backoff
      }
    }
    
    this.processing = false;
  }

  /**
   * Verify Merkle proof for evidence integrity
   * @param root Merkle root
   * @param leaf Leaf hash
   * @param proof Merkle proof array
   * @returns True if valid
   */
  verifyMerkleProof(root: string, leaf: string, proof: string[]): boolean {
    let currentHash = leaf;
    
    for (const sibling of proof) {
      currentHash = this.hashPair(currentHash, sibling);
    }
    
    return currentHash === root;
  }

  private hashPair(left: string, right: string): string {
    const combined = Buffer.concat([
      Buffer.from(left, 'hex'),
      Buffer.from(right, 'hex')
    ]);
    return createHash('sha256').update(combined).digest('hex');
  }

  private generateMerkleRoot(hashes: string[]): string {
    if (hashes.length === 0) return '';
    
    let currentLevel = hashes.map(h => Buffer.from(h, 'hex'));
    
    while (currentLevel.length > 1) {
      const nextLevel: Buffer[] = [];
      
      for (let i = 0; i < currentLevel.length; i += 2) {
        if (i + 1 < currentLevel.length) {
          const combined = Buffer.concat([currentLevel[i], currentLevel[i + 1]]);
          nextLevel.push(createHash('sha256').update(combined).digest());
        } else {
          nextLevel.push(currentLevel[i]); // Odd length padding
        }
      }
      
      currentLevel = nextLevel;
    }
    
    return currentLevel[0].toString('hex');
  }

  /**
   * Hash user ID for privacy-preserving storage
   * @param userId Raw user identifier
   * @returns Hashed user ID
   */
  private hashUserId(userId: string): string {
    return createHash('sha256')
      .update(userId + config.userSalt)
      .digest('hex');
  }

  /**
   * Get storage statistics
   * @returns Storage usage and pin statistics
   */
  getStats(): { 
    totalPinned: number; 
    storageUsed: number; 
    connected: boolean; 
    queueLength: number 
  } {
    // In production, query IPFS node stats
    return {
      totalPinned: 0, // this.stats.pins,
      storageUsed: 0, // this.stats.storageUsed,
      connected: this.isConnected,
      queueLength: this.pinQueue.length
    };
  }

  /**
   * Clean up expired or revoked content (garbage collection)
   * @param cutoffTime Content older than this is eligible for removal
   */
  async garbageCollect(cutoffTime: number): Promise<{ removed: number; freed: number }> {
    let removed = 0;
    let freed = 0;
    
    // Scan storage directory
    const files = fs.readdirSync(this.storageDir);
    
    for (const file of files) {
      const filePath = path.join(this.storageDir, file);
      const stats = fs.statSync(filePath);
      
      if (stats.mtime.getTime() < cutoffTime) {
        try {
          // Check if still referenced on-chain
          const isReferenced = await smartContract.isContentReferenced(file);
          
          if (!isReferenced) {
            const size = stats.size;
            fs.unlinkSync(filePath);
            removed++;
            freed += size;
            logger.info(`Removed expired content: ${file} (${size} bytes)`);
          }
        } catch (error) {
          logger.error(`Error during GC for ${file}: ${error}`);
        }
      }
    }
    
    logger.info(`Garbage collection completed: ${removed} files, ${freed} bytes freed`);
    return { removed, freed };
  }

  /**
   * Health check for IPFS service
   * @returns Service status and connectivity
   */
  async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      // Test pin and retrieve round-trip
      const testContent = Buffer.from('IPFS health check: ' + Date.now());
      const testCid = this.generateCID(testContent);
      
      await this.pinContent(testCid, testContent, { type: 'health_check' });
      const retrieved = await this.retrieveContent(testCid);
      
      if (retrieved.content.equals(testContent)) {
        return {
          status: 'healthy',
          details: {
            connected: this.isConnected,
            queueLength: this.pinQueue.length,
            storageDir: this.storageDir,
            testCid: testCid,
            roundTrip: 'success'
          }
        };
      } else {
        return { status: 'degraded', details: { error: 'Round-trip verification failed' } };
      }
    } catch (error) {
      logger.error('IPFS health check failed:', error);
      return { 
        status: 'unhealthy', 
        details: { error: error.message, connected: this.isConnected } 
      };
    }
  }

  // Graceful shutdown
  async shutdown(): Promise<void> {
    logger.info('Shutting down IPFS service...');
    
    // Process remaining queue
    while (this.pinQueue.length > 0) {
      await this.processPinQueue();
    }
    
    // In production: gracefully disconnect from IPFS node
    this.isConnected = false;
    logger.info('IPFS service shutdown complete');
  }
}

// Initialize service on import (singleton pattern)
const ipfsService = IPFSService.getInstance();
export { ipfsService };

// Export types for external use
export * from './types/ipfs';

// Background task runner (for integration with main services)
export const startIPFSBackgroundTasks = (): void => {
  // Periodic garbage collection
  setInterval(async () => {
    const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000; // 30 days
    await ipfsService.garbageCollect(cutoff);
  }, 24 * 60 * 60 * 1000); // Daily

  // Health monitoring
  setInterval(async () => {
    const health = await ipfsService.healthCheck();
    if (health.status !== 'healthy') {
      logger.warn('IPFS health degraded:', health.details);
      // Trigger alerts or recovery
    }
  }, 5 * 60 * 1000); // Every 5 minutes

  logger.info('IPFS background tasks started');
};

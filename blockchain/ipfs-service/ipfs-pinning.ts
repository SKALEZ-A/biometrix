import { create, IPFSHTTPClient } from 'ipfs-http-client';

export class IPFSPinningService {
  private client: IPFSHTTPClient;
  private pinnedHashes: Set<string>;

  constructor() {
    this.client = create({
      host: process.env.IPFS_HOST || 'localhost',
      port: parseInt(process.env.IPFS_PORT || '5001'),
      protocol: process.env.IPFS_PROTOCOL || 'http'
    });
    this.pinnedHashes = new Set();
  }

  async pinHash(cid: string): Promise<void> {
    try {
      await this.client.pin.add(cid);
      this.pinnedHashes.add(cid);
      console.log(`Successfully pinned: ${cid}`);
    } catch (error) {
      console.error(`Failed to pin ${cid}:`, error);
      throw error;
    }
  }

  async unpinHash(cid: string): Promise<void> {
    try {
      await this.client.pin.rm(cid);
      this.pinnedHashes.delete(cid);
      console.log(`Successfully unpinned: ${cid}`);
    } catch (error) {
      console.error(`Failed to unpin ${cid}:`, error);
      throw error;
    }
  }

  async listPinnedHashes(): Promise<string[]> {
    const pins = [];
    
    for await (const pin of this.client.pin.ls()) {
      pins.push(pin.cid.toString());
    }
    
    return pins;
  }

  async isPinned(cid: string): Promise<boolean> {
    try {
      for await (const pin of this.client.pin.ls({ paths: [cid] })) {
        if (pin.cid.toString() === cid) {
          return true;
        }
      }
      return false;
    } catch (error) {
      return false;
    }
  }

  async pinBiometricData(userId: string, biometricHash: string): Promise<string> {
    const data = {
      userId,
      biometricHash,
      timestamp: new Date().toISOString()
    };

    const { cid } = await this.client.add(JSON.stringify(data));
    await this.pinHash(cid.toString());
    
    return cid.toString();
  }

  async pinFraudReport(reportData: any): Promise<string> {
    const { cid } = await this.client.add(JSON.stringify(reportData));
    await this.pinHash(cid.toString());
    
    return cid.toString();
  }
}

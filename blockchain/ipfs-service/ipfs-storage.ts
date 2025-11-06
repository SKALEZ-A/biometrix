import { IPFSClient } from './ipfs-client';
import { encrypt, decrypt } from '../../packages/shared/src/crypto/encryption';

export interface StorageMetadata {
  cid: string;
  size: number;
  encrypted: boolean;
  timestamp: Date;
  contentType?: string;
}

export class IPFSStorage {
  private ipfsClient: IPFSClient;
  private encryptionKey?: string;

  constructor(ipfsUrl: string, encryptionKey?: string) {
    this.ipfsClient = new IPFSClient(ipfsUrl);
    this.encryptionKey = encryptionKey;
  }

  async store(data: Buffer | string, encrypt: boolean = true): Promise<StorageMetadata> {
    try {
      let content = typeof data === 'string' ? Buffer.from(data) : data;

      if (encrypt && this.encryptionKey) {
        content = Buffer.from(await this.encryptData(content.toString()));
      }

      const cid = await this.ipfsClient.uploadFile(content);
      await this.ipfsClient.pinFile(cid);

      const stats = await this.ipfsClient.getFileStats(cid);

      return {
        cid,
        size: stats.size,
        encrypted: encrypt && !!this.encryptionKey,
        timestamp: new Date()
      };
    } catch (error) {
      throw new Error(`Storage failed: ${error}`);
    }
  }

  async retrieve(cid: string, encrypted: boolean = true): Promise<Buffer> {
    try {
      let content = await this.ipfsClient.downloadFile(cid);

      if (encrypted && this.encryptionKey) {
        const decrypted = await this.decryptData(content.toString());
        content = Buffer.from(decrypted);
      }

      return content;
    } catch (error) {
      throw new Error(`Retrieval failed: ${error}`);
    }
  }

  async delete(cid: string): Promise<void> {
    try {
      await this.ipfsClient.unpinFile(cid);
    } catch (error) {
      throw new Error(`Deletion failed: ${error}`);
    }
  }

  private async encryptData(data: string): Promise<string> {
    if (!this.encryptionKey) {
      throw new Error('Encryption key not provided');
    }
    return encrypt(data, this.encryptionKey);
  }

  private async decryptData(data: string): Promise<string> {
    if (!this.encryptionKey) {
      throw new Error('Encryption key not provided');
    }
    return decrypt(data, this.encryptionKey);
  }
}

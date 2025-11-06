import { create, IPFSHTTPClient } from 'ipfs-http-client';
import { logger } from '@shared/utils/logger';

export class IPFSClient {
  private client: IPFSHTTPClient;

  constructor(url: string = 'http://localhost:5001') {
    this.client = create({ url });
  }

  async add(content: string | Buffer): Promise<string> {
    try {
      const result = await this.client.add(content);
      logger.info('Content added to IPFS', { cid: result.path });
      return result.path;
    } catch (error) {
      logger.error('Failed to add content to IPFS', { error });
      throw error;
    }
  }

  async get(cid: string): Promise<Buffer> {
    try {
      const chunks = [];
      for await (const chunk of this.client.cat(cid)) {
        chunks.push(chunk);
      }
      return Buffer.concat(chunks);
    } catch (error) {
      logger.error('Failed to get content from IPFS', { error, cid });
      throw error;
    }
  }

  async pin(cid: string): Promise<void> {
    try {
      await this.client.pin.add(cid);
      logger.info('Content pinned', { cid });
    } catch (error) {
      logger.error('Failed to pin content', { error, cid });
      throw error;
    }
  }

  async unpin(cid: string): Promise<void> {
    try {
      await this.client.pin.rm(cid);
      logger.info('Content unpinned', { cid });
    } catch (error) {
      logger.error('Failed to unpin content', { error, cid });
      throw error;
    }
  }

  async listPins(): Promise<string[]> {
    try {
      const pins = [];
      for await (const pin of this.client.pin.ls()) {
        pins.push(pin.cid.toString());
      }
      return pins;
    } catch (error) {
      logger.error('Failed to list pins', { error });
      throw error;
    }
  }
}

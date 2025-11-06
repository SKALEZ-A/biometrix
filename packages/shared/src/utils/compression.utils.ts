import { gzip, gunzip } from 'zlib';
import { promisify } from 'util';

const gzipAsync = promisify(gzip);
const gunzipAsync = promisify(gunzip);

export class CompressionUtils {
  static async compress(data: string | Buffer): Promise<Buffer> {
    const input = typeof data === 'string' ? Buffer.from(data) : data;
    return await gzipAsync(input);
  }

  static async decompress(data: Buffer): Promise<Buffer> {
    return await gunzipAsync(data);
  }

  static async compressJSON(obj: any): Promise<Buffer> {
    const json = JSON.stringify(obj);
    return await this.compress(json);
  }

  static async decompressJSON<T = any>(data: Buffer): Promise<T> {
    const decompressed = await this.decompress(data);
    return JSON.parse(decompressed.toString());
  }

  static calculateCompressionRatio(original: Buffer, compressed: Buffer): number {
    return compressed.length / original.length;
  }

  static shouldCompress(data: Buffer, threshold: number = 1024): boolean {
    return data.length > threshold;
  }

  static async compressIfBeneficial(
    data: string | Buffer,
    threshold: number = 1024
  ): Promise<{ compressed: boolean; data: Buffer }> {
    const input = typeof data === 'string' ? Buffer.from(data) : data;
    
    if (!this.shouldCompress(input, threshold)) {
      return { compressed: false, data: input };
    }

    const compressed = await this.compress(input);
    const ratio = this.calculateCompressionRatio(input, compressed);

    if (ratio < 0.9) {
      return { compressed: true, data: compressed };
    }

    return { compressed: false, data: input };
  }
}

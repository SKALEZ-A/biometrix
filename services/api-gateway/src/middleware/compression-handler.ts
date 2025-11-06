import { Request, Response, NextFunction } from 'express';
import * as zlib from 'zlib';

export class CompressionHandler {
  private threshold: number;
  private level: number;

  constructor(threshold = 1024, level = 6) {
    this.threshold = threshold;
    this.level = level;
  }

  public middleware() {
    return (req: Request, res: Response, next: NextFunction) => {
      const acceptEncoding = req.headers['accept-encoding'] || '';
      
      if (!acceptEncoding.match(/\b(gzip|deflate)\b/)) {
        next();
        return;
      }

      const originalWrite = res.write.bind(res);
      const originalEnd = res.end.bind(res);
      let buffer: Buffer[] = [];

      res.write = function(chunk: any, encoding?: any): boolean {
        if (chunk) {
          buffer.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, encoding));
        }
        return true;
      };

      res.end = function(chunk?: any, encoding?: any): Response {
        if (chunk) {
          buffer.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, encoding));
        }

        const body = Buffer.concat(buffer);

        if (body.length < this.threshold) {
          res.write = originalWrite;
          res.end = originalEnd;
          return originalEnd(body);
        }

        const compress = acceptEncoding.includes('gzip') ? zlib.gzip : zlib.deflate;
        const encoding = acceptEncoding.includes('gzip') ? 'gzip' : 'deflate';

        compress(body, { level: this.level }, (err, compressed) => {
          if (err) {
            res.write = originalWrite;
            res.end = originalEnd;
            return originalEnd(body);
          }

          res.setHeader('Content-Encoding', encoding);
          res.setHeader('Content-Length', compressed.length);
          res.removeHeader('Content-Length');
          
          res.write = originalWrite;
          res.end = originalEnd;
          originalEnd(compressed);
        });

        return res;
      }.bind(this);

      next();
    };
  }
}

export const compressionHandler = new CompressionHandler();

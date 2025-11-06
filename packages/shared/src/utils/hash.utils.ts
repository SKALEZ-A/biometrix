import crypto from 'crypto';

export function sha256(data: string | Buffer): string {
  return crypto.createHash('sha256').update(data).digest('hex');
}

export function sha512(data: string | Buffer): string {
  return crypto.createHash('sha512').update(data).digest('hex');
}

export function md5(data: string | Buffer): string {
  return crypto.createHash('md5').update(data).digest('hex');
}

export function hmacSha256(data: string, secret: string): string {
  return crypto.createHmac('sha256', secret).update(data).digest('hex');
}

export function hmacSha512(data: string, secret: string): string {
  return crypto.createHmac('sha512', secret).update(data).digest('hex');
}

export function generateSalt(length: number = 32): string {
  return crypto.randomBytes(length).toString('hex');
}

export function hashPassword(password: string, salt: string): string {
  return crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha512').toString('hex');
}

export function verifyPassword(password: string, salt: string, hash: string): boolean {
  const computedHash = hashPassword(password, salt);
  return crypto.timingSafeEqual(Buffer.from(hash), Buffer.from(computedHash));
}

export function generateRandomToken(length: number = 32): string {
  return crypto.randomBytes(length).toString('base64url');
}

export function generateUUID(): string {
  return crypto.randomUUID();
}

export function hashObject(obj: any): string {
  const str = JSON.stringify(obj, Object.keys(obj).sort());
  return sha256(str);
}

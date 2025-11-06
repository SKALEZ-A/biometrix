import { RedisClient } from '../../../packages/shared/src/cache/redis';
import { Session } from '../models/session.model';

export class SessionRepository {
  private redis: RedisClient;
  private readonly SESSION_PREFIX = 'session:';
  private readonly SESSION_TTL = 86400; // 24 hours

  constructor() {
    this.redis = new RedisClient();
  }

  async createSession(userId: string, token: string, metadata: any): Promise<Session> {
    const session: Session = {
      id: token,
      userId,
      token,
      createdAt: new Date(),
      expiresAt: new Date(Date.now() + this.SESSION_TTL * 1000),
      metadata
    };

    await this.redis.set(
      `${this.SESSION_PREFIX}${token}`,
      JSON.stringify(session),
      this.SESSION_TTL
    );

    return session;
  }

  async getSession(token: string): Promise<Session | null> {
    const data = await this.redis.get(`${this.SESSION_PREFIX}${token}`);
    return data ? JSON.parse(data) : null;
  }

  async deleteSession(token: string): Promise<void> {
    await this.redis.del(`${this.SESSION_PREFIX}${token}`);
  }

  async getUserSessions(userId: string): Promise<Session[]> {
    const keys = await this.redis.keys(`${this.SESSION_PREFIX}*`);
    const sessions: Session[] = [];

    for (const key of keys) {
      const data = await this.redis.get(key);
      if (data) {
        const session = JSON.parse(data);
        if (session.userId === userId) {
          sessions.push(session);
        }
      }
    }

    return sessions;
  }

  async extendSession(token: string): Promise<void> {
    await this.redis.expire(`${this.SESSION_PREFIX}${token}`, this.SESSION_TTL);
  }
}

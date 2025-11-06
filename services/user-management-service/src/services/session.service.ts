import { SessionRepository } from '../repositories/session.repository';
import { Session } from '../models/session.model';
import { generateToken } from '../../../packages/shared/src/crypto/jwt';

export class SessionService {
  private sessionRepository: SessionRepository;

  constructor() {
    this.sessionRepository = new SessionRepository();
  }

  async createSession(userId: string, metadata: any): Promise<Session> {
    const token = generateToken({ userId, type: 'session' });
    return await this.sessionRepository.createSession(userId, token, metadata);
  }

  async validateSession(token: string): Promise<Session | null> {
    const session = await this.sessionRepository.getSession(token);
    
    if (!session) {
      return null;
    }

    if (new Date() > session.expiresAt) {
      await this.sessionRepository.deleteSession(token);
      return null;
    }

    await this.sessionRepository.extendSession(token);
    return session;
  }

  async terminateSession(token: string): Promise<void> {
    await this.sessionRepository.deleteSession(token);
  }

  async terminateAllUserSessions(userId: string): Promise<void> {
    const sessions = await this.sessionRepository.getUserSessions(userId);
    
    for (const session of sessions) {
      await this.sessionRepository.deleteSession(session.token);
    }
  }

  async getUserActiveSessions(userId: string): Promise<Session[]> {
    const sessions = await this.sessionRepository.getUserSessions(userId);
    const now = new Date();
    
    return sessions.filter(session => session.expiresAt > now);
  }
}

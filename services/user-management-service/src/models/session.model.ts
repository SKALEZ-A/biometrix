export interface Session {
  id: string;
  userId: string;
  token: string;
  createdAt: Date;
  expiresAt: Date;
  metadata: {
    ipAddress?: string;
    userAgent?: string;
    deviceId?: string;
    location?: {
      country?: string;
      city?: string;
    };
  };
}

export interface SessionActivity {
  sessionId: string;
  userId: string;
  action: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

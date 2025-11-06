import { Request, Response, NextFunction } from 'express';
import { BiometricProcessor } from '../services/biometric.processor';
import { BehavioralProfileManager } from '../services/profile.manager';
import { KafkaProducerService } from '../services/kafka.producer';
import { MongoDBService } from '../services/mongodb.service';
import { InfluxDBService } from '../services/influxdb.service';

export class BiometricController {
  private biometricProcessor: BiometricProcessor;
  private profileManager: BehavioralProfileManager;
  private kafkaProducer: KafkaProducerService;
  private mongoService: MongoDBService;
  private influxService: InfluxDBService;

  constructor() {
    this.biometricProcessor = new BiometricProcessor();
    this.profileManager = new BehavioralProfileManager();
    this.kafkaProducer = new KafkaProducerService();
    this.mongoService = new MongoDBService();
    this.influxService = new InfluxDBService();
  }

  async ingestEvents(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId, sessionId, events } = req.body;

      if (!userId || !sessionId || !Array.isArray(events)) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing required fields: userId, sessionId, or events'
          }
        });
        return;
      }

      const processedEvents = await this.biometricProcessor.processEvents(events);
      
      await Promise.all([
        this.kafkaProducer.publishEvents('biometric-events', processedEvents),
        this.influxService.writeEvents(userId, sessionId, processedEvents)
      ]);

      res.status(202).json({
        success: true,
        message: 'Events ingested successfully',
        eventCount: events.length,
        sessionId
      });
    } catch (error) {
      next(error);
    }
  }

  async generateProfile(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId, events } = req.body;

      if (!userId || !Array.isArray(events)) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing required fields: userId or events'
          }
        });
        return;
      }

      if (events.length < 500) {
        res.status(400).json({
          error: {
            code: 'INSUFFICIENT_DATA',
            message: 'Minimum 500 events required for profile generation'
          }
        });
        return;
      }

      const keystrokeEvents = events.filter(e => e.type === 'keystroke');
      const mouseEvents = events.filter(e => e.type === 'mouse');
      const touchEvents = events.filter(e => e.type === 'touch');

      const [keystrokeFeatures, mouseFeatures, touchFeatures] = await Promise.all([
        this.biometricProcessor.processKeystrokeEvents(keystrokeEvents),
        this.biometricProcessor.processMouseEvents(mouseEvents),
        this.biometricProcessor.processTouchEvents(touchEvents)
      ]);

      const profile = {
        userId,
        keystroke: keystrokeFeatures,
        mouse: mouseFeatures,
        touch: touchFeatures,
        lastUpdated: new Date(),
        confidence: this.calculateConfidence(events.length),
        sampleCount: events.length,
        version: '1.0'
      };

      await this.mongoService.saveProfile(profile);

      res.status(201).json({
        success: true,
        profile,
        message: 'Behavioral profile generated successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  async getProfile(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId } = req.params;

      if (!userId) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing userId parameter'
          }
        });
        return;
      }

      const profile = await this.mongoService.getProfile(userId);

      if (!profile) {
        res.status(404).json({
          error: {
            code: 'PROFILE_NOT_FOUND',
            message: `No profile found for user ${userId}`
          }
        });
        return;
      }

      res.status(200).json({
        success: true,
        profile
      });
    } catch (error) {
      next(error);
    }
  }

  async verifyBiometric(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId, sessionId, events, zkProof } = req.body;

      if (!userId || !sessionId || !Array.isArray(events)) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing required fields'
          }
        });
        return;
      }

      const storedProfile = await this.mongoService.getProfile(userId);

      if (!storedProfile) {
        res.status(404).json({
          error: {
            code: 'PROFILE_NOT_FOUND',
            message: 'User profile not found'
          }
        });
        return;
      }

      const keystrokeEvents = events.filter(e => e.type === 'keystroke');
      const mouseEvents = events.filter(e => e.type === 'mouse');
      const touchEvents = events.filter(e => e.type === 'touch');

      const [currentKeystroke, currentMouse, currentTouch] = await Promise.all([
        this.biometricProcessor.processKeystrokeEvents(keystrokeEvents),
        this.biometricProcessor.processMouseEvents(mouseEvents),
        this.biometricProcessor.processTouchEvents(touchEvents)
      ]);

      const matchScore = this.profileManager.calculateMatchScore(
        storedProfile,
        { keystroke: currentKeystroke, mouse: currentMouse, touch: currentTouch }
      );

      const anomalyScore = this.profileManager.calculateAnomalyScore(
        storedProfile,
        { keystroke: currentKeystroke, mouse: currentMouse, touch: currentTouch }
      );

      const verified = matchScore >= 0.85 && anomalyScore < 0.3;
      const confidence = this.calculateVerificationConfidence(matchScore, anomalyScore);

      const reasons = [];
      if (matchScore < 0.85) {
        reasons.push('Behavioral pattern mismatch detected');
      }
      if (anomalyScore >= 0.3) {
        reasons.push('Anomalous behavior detected');
      }

      if (verified) {
        await this.profileManager.updateProfile(userId, {
          keystroke: currentKeystroke,
          mouse: currentMouse,
          touch: currentTouch
        });
      }

      res.status(200).json({
        verified,
        confidence,
        matchScore,
        anomalyScore,
        reasons
      });
    } catch (error) {
      next(error);
    }
  }

  async updateProfile(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId } = req.params;
      const { events } = req.body;

      if (!userId || !Array.isArray(events)) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing required fields'
          }
        });
        return;
      }

      const storedProfile = await this.mongoService.getProfile(userId);

      if (!storedProfile) {
        res.status(404).json({
          error: {
            code: 'PROFILE_NOT_FOUND',
            message: 'User profile not found'
          }
        });
        return;
      }

      const keystrokeEvents = events.filter(e => e.type === 'keystroke');
      const mouseEvents = events.filter(e => e.type === 'mouse');
      const touchEvents = events.filter(e => e.type === 'touch');

      const [newKeystroke, newMouse, newTouch] = await Promise.all([
        this.biometricProcessor.processKeystrokeEvents(keystrokeEvents),
        this.biometricProcessor.processMouseEvents(mouseEvents),
        this.biometricProcessor.processTouchEvents(touchEvents)
      ]);

      await this.profileManager.updateProfile(userId, {
        keystroke: newKeystroke,
        mouse: newMouse,
        touch: newTouch
      });

      res.status(200).json({
        success: true,
        message: 'Profile updated successfully'
      });
    } catch (error) {
      next(error);
    }
  }

  async getHistoricalMetrics(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      const { userId } = req.params;
      const { startTime, endTime, metric } = req.query;

      if (!userId) {
        res.status(400).json({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Missing userId parameter'
          }
        });
        return;
      }

      const metrics = await this.influxService.queryMetrics(
        userId,
        startTime as string,
        endTime as string,
        metric as string
      );

      res.status(200).json({
        success: true,
        metrics,
        count: metrics.length
      });
    } catch (error) {
      next(error);
    }
  }

  private calculateConfidence(sampleCount: number): number {
    const minSamples = 500;
    const maxSamples = 5000;
    
    if (sampleCount < minSamples) return 0;
    if (sampleCount >= maxSamples) return 1;
    
    return (sampleCount - minSamples) / (maxSamples - minSamples);
  }

  private calculateVerificationConfidence(matchScore: number, anomalyScore: number): number {
    const matchWeight = 0.7;
    const anomalyWeight = 0.3;
    
    return matchScore * matchWeight + (1 - anomalyScore) * anomalyWeight;
  }
}

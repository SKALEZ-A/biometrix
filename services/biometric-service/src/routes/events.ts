import { Router, Request, Response, NextFunction } from 'express';
import { validateEventSchema } from '../validators/eventValidator';
import { biometricProcessor } from '../processors/biometricProcessor';
import { logger } from '../utils/logger';
import { EventType, BiometricEvent } from '../types/biometric';

const router = Router();

// POST /api/v1/biometric/events - Ingest biometric events
router.post('/', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { userId, sessionId, events } = req.body;

    // Validate input
    const validation = validateEventSchema({ userId, sessionId, events });
    if (!validation.valid) {
      return res.status(400).json({ error: 'Invalid event data', details: validation.errors });
    }

    // Process events in parallel
    const processedEvents = await Promise.all(
      events.map(async (event: BiometricEvent) => {
        await biometricProcessor.processEvent(userId, sessionId, event);
        return event;
      })
    );

    // Calculate session risk score
    const riskScore = biometricProcessor.calculateSessionRisk(userId, sessionId);
    
    logger.info(`Processed ${events.length} events for user ${userId}, risk: ${riskScore}`);

    res.status(200).json({
      success: true,
      processed: processedEvents.length,
      riskScore,
      sessionId,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error processing events: ${error}`);
    next(error);
  }
});

// GET /api/v1/biometric/events/:sessionId - Retrieve session events
router.get('/:sessionId', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { sessionId } = req.params;
    const { userId, limit = 1000, type }: { userId?: string; limit?: number; type?: EventType } = req.query;

    const events = await biometricProcessor.retrieveEvents(userId, sessionId, { limit: parseInt(limit as string), type: type as EventType });
    
    res.status(200).json({
      sessionId,
      events,
      count: events.length,
      filters: { userId, limit, type }
    });
  } catch (error) {
    logger.error(`Error retrieving events: ${error}`);
    next(error);
  }
});

// DELETE /api/v1/biometric/events/:sessionId - Clear session events (GDPR compliance)
router.delete('/:sessionId', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { sessionId } = req.params;
    const { userId, reason } = req.body; // reason: 'gdpr_request' | 'session_end'

    await biometricProcessor.deleteEvents(userId, sessionId, reason);
    
    logger.info(`Deleted events for session ${sessionId} due to ${reason}`);
    
    res.status(200).json({ success: true, sessionId, action: 'deleted', reason });
  } catch (error) {
    logger.error(`Error deleting events: ${error}`);
    next(error);
  }
});

export default router;

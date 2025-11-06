import { Request, Response } from 'express';
import { voiceprintService } from '../services/voiceprint.service';
import { logger } from '../../../packages/shared/src/utils/logger';

export class VoiceController {
  public async enrollVoiceprint(req: Request, res: Response): Promise<void> {
    try {
      const { userId, audioData } = req.body;

      if (!userId || !audioData) {
        res.status(400).json({
          error: 'Bad Request',
          message: 'userId and audioData are required'
        });
        return;
      }

      const result = await voiceprintService.enrollVoiceprint(userId, audioData);

      res.status(201).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error enrolling voiceprint:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to enroll voiceprint'
      });
    }
  }

  public async verifyVoice(req: Request, res: Response): Promise<void> {
    try {
      const { userId, audioData } = req.body;

      if (!userId || !audioData) {
        res.status(400).json({
          error: 'Bad Request',
          message: 'userId and audioData are required'
        });
        return;
      }

      const result = await voiceprintService.verifyVoice(userId, audioData);

      res.status(200).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error verifying voice:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to verify voice'
      });
    }
  }

  public async getVoiceprint(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;

      const voiceprint = await voiceprintService.getVoiceprint(userId);

      if (!voiceprint) {
        res.status(404).json({
          error: 'Not Found',
          message: 'Voiceprint not found for this user'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: voiceprint
      });
    } catch (error) {
      logger.error('Error getting voiceprint:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to retrieve voiceprint'
      });
    }
  }

  public async deleteVoiceprint(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;

      const deleted = await voiceprintService.deleteVoiceprint(userId);

      if (!deleted) {
        res.status(404).json({
          error: 'Not Found',
          message: 'Voiceprint not found for this user'
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: 'Voiceprint deleted successfully'
      });
    } catch (error) {
      logger.error('Error deleting voiceprint:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to delete voiceprint'
      });
    }
  }

  public async updateVoiceprint(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.params;
      const { audioData } = req.body;

      if (!audioData) {
        res.status(400).json({
          error: 'Bad Request',
          message: 'audioData is required'
        });
        return;
      }

      const result = await voiceprintService.updateVoiceprint(userId, audioData);

      res.status(200).json({
        success: true,
        data: result
      });
    } catch (error) {
      logger.error('Error updating voiceprint:', error);
      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Failed to update voiceprint'
      });
    }
  }
}

export const voiceController = new VoiceController();

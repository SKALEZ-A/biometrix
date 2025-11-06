import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { BiometricController } from '../src/controllers/biometric.controller';
import { Request, Response } from 'express';

describe('BiometricController', () => {
  let controller: BiometricController;
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;

  beforeEach(() => {
    controller = new BiometricController();
    mockRequest = {
      body: {},
      params: {},
      user: { id: 'test-user-id', email: 'test@example.com', role: 'user' },
    };
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };
  });

  describe('enrollBiometric', () => {
    it('should enroll fingerprint biometric successfully', async () => {
      mockRequest.body = {
        userId: 'test-user-id',
        biometricType: 'fingerprint',
        template: 'base64-encoded-template',
      };

      await controller.enrollBiometric(
        mockRequest as Request,
        mockResponse as Response
      );

      expect(mockResponse.status).toHaveBeenCalledWith(201);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          success: true,
          message: 'Biometric enrolled successfully',
        })
      );
    });

    it('should return 400 for invalid biometric type', async () => {
      mockRequest.body = {
        userId: 'test-user-id',
        biometricType: 'invalid-type',
        template: 'base64-encoded-template',
      };

      await controller.enrollBiometric(
        mockRequest as Request,
        mockResponse as Response
      );

      expect(mockResponse.status).toHaveBeenCalledWith(400);
    });
  });

  describe('verifyBiometric', () => {
    it('should verify biometric successfully', async () => {
      mockRequest.body = {
        userId: 'test-user-id',
        biometricType: 'fingerprint',
        template: 'base64-encoded-template',
      };

      await controller.verifyBiometric(
        mockRequest as Request,
        mockResponse as Response
      );

      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          match: expect.any(Boolean),
          confidence: expect.any(Number),
        })
      );
    });
  });

  describe('getBiometricProfile', () => {
    it('should retrieve biometric profile', async () => {
      mockRequest.params = { userId: 'test-user-id' };

      await controller.getBiometricProfile(
        mockRequest as Request,
        mockResponse as Response
      );

      expect(mockResponse.status).toHaveBeenCalledWith(200);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: 'test-user-id',
          biometrics: expect.any(Array),
        })
      );
    });
  });
});

import { VoiceAuthenticationService } from '../src/services/voice-authentication.service';

describe('VoiceAuthenticationService', () => {
  let service: VoiceAuthenticationService;

  beforeEach(() => {
    service = new VoiceAuthenticationService();
  });

  describe('enrollVoiceprint', () => {
    it('should enroll voice sample', async () => {
      const audioBuffer = Buffer.from('mock-audio-data');
      const result = await service.enrollVoiceprint('user123', audioBuffer);
      
      expect(result).toHaveProperty('voiceprintId');
      expect(result).toHaveProperty('quality');
      expect(result.quality).toBeGreaterThan(0);
    });

    it('should reject poor quality audio', async () => {
      const poorAudio = Buffer.from('bad');
      await expect(service.enrollVoiceprint('user123', poorAudio)).rejects.toThrow();
    });
  });

  describe('verifyVoice', () => {
    it('should verify voice against enrolled voiceprint', async () => {
      const audioBuffer = Buffer.from('mock-audio-data');
      await service.enrollVoiceprint('user123', audioBuffer);
      
      const result = await service.verifyVoice('user123', audioBuffer);
      expect(result).toHaveProperty('verified');
      expect(result).toHaveProperty('confidence');
    });

    it('should return false for non-matching voice', async () => {
      const enrollAudio = Buffer.from('voice-1');
      const verifyAudio = Buffer.from('voice-2');
      
      await service.enrollVoiceprint('user123', enrollAudio);
      const result = await service.verifyVoice('user123', verifyAudio);
      
      expect(result.verified).toBe(false);
    });
  });

  describe('extractVoiceFeatures', () => {
    it('should extract voice features', async () => {
      const audioBuffer = Buffer.from('mock-audio-data');
      const features = await service.extractVoiceFeatures(audioBuffer);
      
      expect(features).toHaveProperty('mfcc');
      expect(features).toHaveProperty('pitch');
      expect(features).toHaveProperty('energy');
    });
  });
});

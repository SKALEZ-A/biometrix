import { z } from 'zod';

export const enrollVoiceprintSchema = z.object({
  userId: z.string().min(1, 'User ID is required'),
  audioData: z.string().min(1, 'Audio data is required'),
  metadata: z.record(z.any()).optional()
});

export const verifyVoiceSchema = z.object({
  userId: z.string().min(1, 'User ID is required'),
  audioData: z.string().min(1, 'Audio data is required')
});

export const updateVoiceprintSchema = z.object({
  audioData: z.string().min(1, 'Audio data is required'),
  metadata: z.record(z.any()).optional()
});

export type EnrollVoiceprintInput = z.infer<typeof enrollVoiceprintSchema>;
export type VerifyVoiceInput = z.infer<typeof verifyVoiceSchema>;
export type UpdateVoiceprintInput = z.infer<typeof updateVoiceprintSchema>;

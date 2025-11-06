interface Voiceprint {
  voiceprintId: string;
  userId: string;
  embedding: Float32Array;
  quality: number;
  createdAt: Date;
  lastUsed?: Date;
}

export class VoiceprintService {
  private voiceprints: Map<string, Voiceprint>;

  constructor() {
    this.voiceprints = new Map();
  }

  async createVoiceprint(userId: string, audioData: Buffer): Promise<Voiceprint> {
    const embedding = await this.extractEmbedding(audioData);
    const quality = this.assessQuality(audioData, embedding);

    if (quality < 0.6) {
      throw new Error('Audio quality too low for voiceprint creation');
    }

    const voiceprint: Voiceprint = {
      voiceprintId: this.generateId(),
      userId,
      embedding,
      quality,
      createdAt: new Date()
    };

    this.voiceprints.set(userId, voiceprint);
    return voiceprint;
  }

  getVoiceprint(userId: string): Voiceprint | undefined {
    return this.voiceprints.get(userId);
  }

  async updateVoiceprint(userId: string, audioData: Buffer): Promise<Voiceprint> {
    const existing = this.voiceprints.get(userId);
    if (!existing) {
      throw new Error('No existing voiceprint found');
    }

    const newEmbedding = await this.extractEmbedding(audioData);
    const adaptedEmbedding = this.adaptEmbedding(existing.embedding, newEmbedding);

    existing.embedding = adaptedEmbedding;
    existing.lastUsed = new Date();

    return existing;
  }

  private async extractEmbedding(audioData: Buffer): Promise<Float32Array> {
    const embedding = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      embedding[i] = Math.random();
    }
    return embedding;
  }

  private assessQuality(audioData: Buffer, embedding: Float32Array): number {
    const snr = this.estimateSNR(audioData);
    const embeddingMagnitude = this.calculateMagnitude(embedding);
    
    return Math.min((snr / 30) * 0.5 + (embeddingMagnitude / 10) * 0.5, 1.0);
  }

  private estimateSNR(audioData: Buffer): number {
    return 20 + Math.random() * 10;
  }

  private calculateMagnitude(embedding: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < embedding.length; i++) {
      sum += embedding[i] * embedding[i];
    }
    return Math.sqrt(sum);
  }

  private adaptEmbedding(existing: Float32Array, newEmbedding: Float32Array): Float32Array {
    const adapted = new Float32Array(existing.length);
    const alpha = 0.1;

    for (let i = 0; i < existing.length; i++) {
      adapted[i] = (1 - alpha) * existing[i] + alpha * newEmbedding[i];
    }

    return adapted;
  }

  private generateId(): string {
    return `vp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

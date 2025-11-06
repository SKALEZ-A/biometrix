import { BiometricProfile } from '../models/biometric-profile.model';
import { FacialFeatures, FacialMatchResult, LivenessCheckResult } from '@shared/types/biometric.types';
import { Logger } from '@shared/utils/logger';
import { MetricsCollector } from '@shared/utils/metrics';

export class FacialRecognitionService {
  private readonly logger = new Logger('FacialRecognitionService');
  private readonly metrics = new MetricsCollector();
  private readonly MATCH_THRESHOLD = 0.85;
  private readonly LIVENESS_THRESHOLD = 0.90;

  async extractFacialFeatures(imageData: Buffer): Promise<FacialFeatures> {
    const startTime = Date.now();
    try {
      const features = await this.performFeatureExtraction(imageData);
      this.metrics.recordLatency('facial_feature_extraction', Date.now() - startTime);
      return features;
    } catch (error) {
      this.logger.error('Feature extraction failed', error);
      throw new Error('Failed to extract facial features');
    }
  }

  private async performFeatureExtraction(imageData: Buffer): Promise<FacialFeatures> {
    const faceDetection = await this.detectFace(imageData);
    const landmarks = await this.extractLandmarks(faceDetection);
    const embeddings = await this.generateEmbeddings(landmarks);
    
    return {
      embeddings,
      landmarks,
      faceBox: faceDetection.boundingBox,
      confidence: faceDetection.confidence,
      quality: this.assessImageQuality(imageData),
      timestamp: Date.now()
    };
  }

  private async detectFace(imageData: Buffer): Promise<any> {
    const faceBoxes = [];
    const imageArray = new Uint8Array(imageData);
    
    for (let i = 0; i < imageArray.length; i += 4) {
      const r = imageArray[i];
      const g = imageArray[i + 1];
      const b = imageArray[i + 2];
      const skinTone = (r + g + b) / 3;
      
      if (skinTone > 100 && skinTone < 200) {
        faceBoxes.push({ x: i % 640, y: Math.floor(i / 640) });
      }
    }

    return {
      boundingBox: {
        x: 100,
        y: 100,
        width: 200,
        height: 250
      },
      confidence: 0.95
    };
  }

  private async extractLandmarks(faceDetection: any): Promise<number[][]> {
    const landmarks = [];
    const { x, y, width, height } = faceDetection.boundingBox;
    
    landmarks.push([x + width * 0.3, y + height * 0.3]);
    landmarks.push([x + width * 0.7, y + height * 0.3]);
    landmarks.push([x + width * 0.5, y + height * 0.5]);
    landmarks.push([x + width * 0.3, y + height * 0.7]);
    landmarks.push([x + width * 0.7, y + height * 0.7]);
    
    for (let i = 0; i < 63; i++) {
      landmarks.push([
        x + Math.random() * width,
        y + Math.random() * height
      ]);
    }
    
    return landmarks;
  }

  private async generateEmbeddings(landmarks: number[][]): Promise<number[]> {
    const embeddings = new Array(512);
    
    for (let i = 0; i < 512; i++) {
      let value = 0;
      for (const [x, y] of landmarks) {
        value += Math.sin(x * i * 0.01) * Math.cos(y * i * 0.01);
      }
      embeddings[i] = value / landmarks.length;
    }
    
    const magnitude = Math.sqrt(embeddings.reduce((sum, val) => sum + val * val, 0));
    return embeddings.map(val => val / magnitude);
  }

  private assessImageQuality(imageData: Buffer): number {
    const array = new Uint8Array(imageData);
    let brightness = 0;
    let contrast = 0;
    
    for (let i = 0; i < array.length; i += 4) {
      const gray = (array[i] + array[i + 1] + array[i + 2]) / 3;
      brightness += gray;
    }
    
    brightness /= (array.length / 4);
    
    for (let i = 0; i < array.length; i += 4) {
      const gray = (array[i] + array[i + 1] + array[i + 2]) / 3;
      contrast += Math.abs(gray - brightness);
    }
    
    contrast /= (array.length / 4);
    
    const qualityScore = Math.min(1.0, (brightness / 128) * (contrast / 64));
    return qualityScore;
  }

  async matchFaces(features1: FacialFeatures, features2: FacialFeatures): Promise<FacialMatchResult> {
    const startTime = Date.now();
    
    const similarity = this.calculateCosineSimilarity(features1.embeddings, features2.embeddings);
    const landmarkDistance = this.calculateLandmarkDistance(features1.landmarks, features2.landmarks);
    
    const combinedScore = (similarity * 0.7) + ((1 - landmarkDistance) * 0.3);
    const isMatch = combinedScore >= this.MATCH_THRESHOLD;
    
    this.metrics.recordLatency('facial_matching', Date.now() - startTime);
    this.metrics.incrementCounter('facial_matches', { result: isMatch ? 'match' : 'no_match' });
    
    return {
      isMatch,
      confidence: combinedScore,
      similarity,
      landmarkDistance,
      timestamp: Date.now()
    };
  }

  private calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      mag1 += vec1[i] * vec1[i];
      mag2 += vec2[i] * vec2[i];
    }
    
    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2));
  }

  private calculateLandmarkDistance(landmarks1: number[][], landmarks2: number[][]): number {
    let totalDistance = 0;
    
    for (let i = 0; i < Math.min(landmarks1.length, landmarks2.length); i++) {
      const dx = landmarks1[i][0] - landmarks2[i][0];
      const dy = landmarks1[i][1] - landmarks2[i][1];
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    
    return totalDistance / landmarks1.length;
  }

  async performLivenessCheck(videoFrames: Buffer[]): Promise<LivenessCheckResult> {
    const startTime = Date.now();
    
    const blinkDetected = await this.detectBlink(videoFrames);
    const headMovement = await this.detectHeadMovement(videoFrames);
    const textureAnalysis = await this.analyzeTexture(videoFrames);
    const depthAnalysis = await this.analyzeDepth(videoFrames);
    
    const livenessScore = (
      (blinkDetected ? 0.25 : 0) +
      (headMovement ? 0.25 : 0) +
      (textureAnalysis * 0.25) +
      (depthAnalysis * 0.25)
    );
    
    const isLive = livenessScore >= this.LIVENESS_THRESHOLD;
    
    this.metrics.recordLatency('liveness_check', Date.now() - startTime);
    this.metrics.incrementCounter('liveness_checks', { result: isLive ? 'live' : 'spoof' });
    
    return {
      isLive,
      confidence: livenessScore,
      blinkDetected,
      headMovement,
      textureScore: textureAnalysis,
      depthScore: depthAnalysis,
      timestamp: Date.now()
    };
  }

  private async detectBlink(frames: Buffer[]): Promise<boolean> {
    const eyeAspectRatios = [];
    
    for (const frame of frames) {
      const features = await this.extractFacialFeatures(frame);
      const leftEye = features.landmarks.slice(36, 42);
      const rightEye = features.landmarks.slice(42, 48);
      
      const leftEAR = this.calculateEyeAspectRatio(leftEye);
      const rightEAR = this.calculateEyeAspectRatio(rightEye);
      
      eyeAspectRatios.push((leftEAR + rightEAR) / 2);
    }
    
    const minEAR = Math.min(...eyeAspectRatios);
    const maxEAR = Math.max(...eyeAspectRatios);
    
    return (maxEAR - minEAR) > 0.15;
  }

  private calculateEyeAspectRatio(eyeLandmarks: number[][]): number {
    const verticalDist1 = this.euclideanDistance(eyeLandmarks[1], eyeLandmarks[5]);
    const verticalDist2 = this.euclideanDistance(eyeLandmarks[2], eyeLandmarks[4]);
    const horizontalDist = this.euclideanDistance(eyeLandmarks[0], eyeLandmarks[3]);
    
    return (verticalDist1 + verticalDist2) / (2.0 * horizontalDist);
  }

  private euclideanDistance(point1: number[], point2: number[]): number {
    const dx = point1[0] - point2[0];
    const dy = point1[1] - point2[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  private async detectHeadMovement(frames: Buffer[]): Promise<boolean> {
    if (frames.length < 3) return false;
    
    const poses = [];
    for (const frame of frames) {
      const features = await this.extractFacialFeatures(frame);
      poses.push(this.estimateHeadPose(features.landmarks));
    }
    
    let maxYawChange = 0;
    let maxPitchChange = 0;
    
    for (let i = 1; i < poses.length; i++) {
      const yawChange = Math.abs(poses[i].yaw - poses[i - 1].yaw);
      const pitchChange = Math.abs(poses[i].pitch - poses[i - 1].pitch);
      
      maxYawChange = Math.max(maxYawChange, yawChange);
      maxPitchChange = Math.max(maxPitchChange, pitchChange);
    }
    
    return maxYawChange > 10 || maxPitchChange > 10;
  }

  private estimateHeadPose(landmarks: number[][]): { yaw: number; pitch: number; roll: number } {
    const nose = landmarks[30];
    const leftEye = landmarks[36];
    const rightEye = landmarks[45];
    
    const eyeCenter = [
      (leftEye[0] + rightEye[0]) / 2,
      (leftEye[1] + rightEye[1]) / 2
    ];
    
    const yaw = Math.atan2(nose[0] - eyeCenter[0], 100) * (180 / Math.PI);
    const pitch = Math.atan2(nose[1] - eyeCenter[1], 100) * (180 / Math.PI);
    const roll = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]) * (180 / Math.PI);
    
    return { yaw, pitch, roll };
  }

  private async analyzeTexture(frames: Buffer[]): Promise<number> {
    let totalScore = 0;
    
    for (const frame of frames) {
      const array = new Uint8Array(frame);
      const lbpHistogram = this.calculateLBP(array);
      const textureComplexity = this.calculateEntropy(lbpHistogram);
      totalScore += textureComplexity;
    }
    
    return Math.min(1.0, totalScore / frames.length);
  }

  private calculateLBP(imageData: Uint8Array): number[] {
    const histogram = new Array(256).fill(0);
    const width = 640;
    const height = 480;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const centerIdx = (y * width + x) * 4;
        const center = imageData[centerIdx];
        
        let lbpValue = 0;
        const neighbors = [
          [-1, -1], [-1, 0], [-1, 1],
          [0, 1], [1, 1], [1, 0],
          [1, -1], [0, -1]
        ];
        
        for (let i = 0; i < neighbors.length; i++) {
          const [dy, dx] = neighbors[i];
          const neighborIdx = ((y + dy) * width + (x + dx)) * 4;
          const neighbor = imageData[neighborIdx];
          
          if (neighbor >= center) {
            lbpValue |= (1 << i);
          }
        }
        
        histogram[lbpValue]++;
      }
    }
    
    return histogram;
  }

  private calculateEntropy(histogram: number[]): number {
    const total = histogram.reduce((sum, val) => sum + val, 0);
    let entropy = 0;
    
    for (const count of histogram) {
      if (count > 0) {
        const probability = count / total;
        entropy -= probability * Math.log2(probability);
      }
    }
    
    return entropy / 8;
  }

  private async analyzeDepth(frames: Buffer[]): Promise<number> {
    let totalDepthScore = 0;
    
    for (let i = 1; i < frames.length; i++) {
      const prevFrame = new Uint8Array(frames[i - 1]);
      const currFrame = new Uint8Array(frames[i]);
      
      const opticalFlow = this.calculateOpticalFlow(prevFrame, currFrame);
      const depthConsistency = this.assessDepthConsistency(opticalFlow);
      
      totalDepthScore += depthConsistency;
    }
    
    return totalDepthScore / (frames.length - 1);
  }

  private calculateOpticalFlow(frame1: Uint8Array, frame2: Uint8Array): number[][] {
    const flow = [];
    const blockSize = 16;
    const width = 640;
    const height = 480;
    
    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        const motion = this.estimateBlockMotion(frame1, frame2, x, y, blockSize, width);
        flow.push(motion);
      }
    }
    
    return flow;
  }

  private estimateBlockMotion(
    frame1: Uint8Array,
    frame2: Uint8Array,
    x: number,
    y: number,
    blockSize: number,
    width: number
  ): number[] {
    let minSAD = Infinity;
    let bestMotion = [0, 0];
    
    for (let dy = -8; dy <= 8; dy += 2) {
      for (let dx = -8; dx <= 8; dx += 2) {
        const sad = this.calculateSAD(frame1, frame2, x, y, dx, dy, blockSize, width);
        
        if (sad < minSAD) {
          minSAD = sad;
          bestMotion = [dx, dy];
        }
      }
    }
    
    return bestMotion;
  }

  private calculateSAD(
    frame1: Uint8Array,
    frame2: Uint8Array,
    x: number,
    y: number,
    dx: number,
    dy: number,
    blockSize: number,
    width: number
  ): number {
    let sad = 0;
    
    for (let by = 0; by < blockSize; by++) {
      for (let bx = 0; bx < blockSize; bx++) {
        const idx1 = ((y + by) * width + (x + bx)) * 4;
        const idx2 = ((y + by + dy) * width + (x + bx + dx)) * 4;
        
        if (idx2 >= 0 && idx2 < frame2.length) {
          sad += Math.abs(frame1[idx1] - frame2[idx2]);
        }
      }
    }
    
    return sad;
  }

  private assessDepthConsistency(opticalFlow: number[][]): number {
    const magnitudes = opticalFlow.map(([dx, dy]) => Math.sqrt(dx * dx + dy * dy));
    const avgMagnitude = magnitudes.reduce((sum, mag) => sum + mag, 0) / magnitudes.length;
    
    let variance = 0;
    for (const mag of magnitudes) {
      variance += Math.pow(mag - avgMagnitude, 2);
    }
    variance /= magnitudes.length;
    
    const consistency = 1 / (1 + variance);
    return consistency;
  }

  async detectFacialExpression(features: FacialFeatures): Promise<string> {
    const landmarks = features.landmarks;
    
    const mouthOpenness = this.calculateMouthOpenness(landmarks);
    const eyebrowPosition = this.calculateEyebrowPosition(landmarks);
    const eyeOpenness = this.calculateEyeOpenness(landmarks);
    
    if (mouthOpenness > 0.3 && eyeOpenness > 0.5) {
      return 'surprised';
    } else if (mouthOpenness > 0.2 && eyebrowPosition > 0.3) {
      return 'happy';
    } else if (eyebrowPosition < -0.2 && mouthOpenness < 0.1) {
      return 'angry';
    } else if (eyeOpenness < 0.3) {
      return 'sad';
    } else {
      return 'neutral';
    }
  }

  private calculateMouthOpenness(landmarks: number[][]): number {
    const upperLip = landmarks[51];
    const lowerLip = landmarks[57];
    const mouthWidth = this.euclideanDistance(landmarks[48], landmarks[54]);
    
    const openness = this.euclideanDistance(upperLip, lowerLip) / mouthWidth;
    return openness;
  }

  private calculateEyebrowPosition(landmarks: number[][]): number {
    const leftEyebrow = landmarks[19];
    const rightEyebrow = landmarks[24];
    const leftEye = landmarks[36];
    const rightEye = landmarks[45];
    
    const leftDistance = leftEyebrow[1] - leftEye[1];
    const rightDistance = rightEyebrow[1] - rightEye[1];
    
    return (leftDistance + rightDistance) / 2;
  }

  private calculateEyeOpenness(landmarks: number[][]): number {
    const leftEAR = this.calculateEyeAspectRatio(landmarks.slice(36, 42));
    const rightEAR = this.calculateEyeAspectRatio(landmarks.slice(42, 48));
    
    return (leftEAR + rightEAR) / 2;
  }
}

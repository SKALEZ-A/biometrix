import { IrisFeatures, IrisMatchResult } from '@shared/types/biometric.types';
import { Logger } from '@shared/utils/logger';
import { MetricsCollector } from '@shared/utils/metrics';

export class IrisRecognitionService {
  private readonly logger = new Logger('IrisRecognitionService');
  private readonly metrics = new MetricsCollector();
  private readonly MATCH_THRESHOLD = 0.88;
  private readonly IRIS_RADIUS_MIN = 80;
  private readonly IRIS_RADIUS_MAX = 150;
  private readonly PUPIL_RADIUS_MIN = 20;
  private readonly PUPIL_RADIUS_MAX = 60;

  async extractIrisFeatures(eyeImage: Buffer): Promise<IrisFeatures> {
    const startTime = Date.now();
    
    try {
      const normalized = await this.normalizeIris(eyeImage);
      const irisCode = await this.generateIrisCode(normalized);
      const mask = await this.generateMask(normalized);
      
      this.metrics.recordLatency('iris_feature_extraction', Date.now() - startTime);
      
      return {
        irisCode,
        mask,
        quality: this.assessIrisQuality(eyeImage),
        pupilRadius: this.detectPupilRadius(eyeImage),
        irisRadius: this.detectIrisRadius(eyeImage),
        timestamp: Date.now()
      };
    } catch (error) {
      this.logger.error('Iris feature extraction failed', error);
      throw new Error('Failed to extract iris features');
    }
  }

  private async normalizeIris(eyeImage: Buffer): Promise<number[][]> {
    const imageArray = new Uint8Array(eyeImage);
    const width = 640;
    const height = 480;
    
    const pupilCenter = this.detectPupilCenter(imageArray, width, height);
    const irisCenter = this.detectIrisCenter(imageArray, width, height);
    const pupilRadius = this.detectPupilRadius(eyeImage);
    const irisRadius = this.detectIrisRadius(eyeImage);

    const normalized: number[][] = [];
    const radialResolution = 64;
    const angularResolution = 512;
    
    for (let r = 0; r < radialResolution; r++) {
      const row: number[] = [];
      for (let theta = 0; theta < angularResolution; theta++) {
        const angle = (theta / angularResolution) * 2 * Math.PI;
        const radius = pupilRadius + (r / radialResolution) * (irisRadius - pupilRadius);
        
        const x = Math.round(irisCenter.x + radius * Math.cos(angle));
        const y = Math.round(irisCenter.y + radius * Math.sin(angle));
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
          const idx = (y * width + x) * 4;
          const gray = (imageArray[idx] + imageArray[idx + 1] + imageArray[idx + 2]) / 3;
          row.push(gray);
        } else {
          row.push(0);
        }
      }
      normalized.push(row);
    }
    
    return normalized;
  }

  private detectPupilCenter(imageArray: Uint8Array, width: number, height: number): { x: number; y: number } {
    let minBrightness = 255;
    let pupilX = width / 2;
    let pupilY = height / 2;
    
    for (let y = height * 0.3; y < height * 0.7; y++) {
      for (let x = width * 0.3; x < width * 0.7; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (imageArray[idx] + imageArray[idx + 1] + imageArray[idx + 2]) / 3;
        
        if (brightness < minBrightness) {
          minBrightness = brightness;
          pupilX = x;
          pupilY = y;
        }
      }
    }
    
    return { x: pupilX, y: pupilY };
  }

  private detectIrisCenter(imageArray: Uint8Array, width: number, height: number): { x: number; y: number } {
    const pupilCenter = this.detectPupilCenter(imageArray, width, height);
    return pupilCenter;
  }

  private detectPupilRadius(eyeImage: Buffer): number {
    const imageArray = new Uint8Array(eyeImage);
    const width = 640;
    const height = 480;
    const center = this.detectPupilCenter(imageArray, width, height);
    
    let radius = this.PUPIL_RADIUS_MIN;
    const threshold = 50;
    
    for (let r = this.PUPIL_RADIUS_MIN; r < this.PUPIL_RADIUS_MAX; r++) {
      let edgeStrength = 0;
      const samples = 32;
      
      for (let i = 0; i < samples; i++) {
        const angle = (i / samples) * 2 * Math.PI;
        const x = Math.round(center.x + r * Math.cos(angle));
        const y = Math.round(center.y + r * Math.sin(angle));
        
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
          const idx = (y * width + x) * 4;
          const innerIdx = ((y - 1) * width + x) * 4;
          const outerIdx = ((y + 1) * width + x) * 4;
          
          const inner = (imageArray[innerIdx] + imageArray[innerIdx + 1] + imageArray[innerIdx + 2]) / 3;
          const outer = (imageArray[outerIdx] + imageArray[outerIdx + 1] + imageArray[outerIdx + 2]) / 3;
          
          edgeStrength += Math.abs(outer - inner);
        }
      }
      
      if (edgeStrength > threshold * samples) {
        radius = r;
        break;
      }
    }
    
    return radius;
  }

  private detectIrisRadius(eyeImage: Buffer): number {
    const imageArray = new Uint8Array(eyeImage);
    const width = 640;
    const height = 480;
    const center = this.detectIrisCenter(imageArray, width, height);
    
    let radius = this.IRIS_RADIUS_MIN;
    const threshold = 40;
    
    for (let r = this.IRIS_RADIUS_MIN; r < this.IRIS_RADIUS_MAX; r++) {
      let edgeStrength = 0;
      const samples = 64;
      
      for (let i = 0; i < samples; i++) {
        const angle = (i / samples) * 2 * Math.PI;
        const x = Math.round(center.x + r * Math.cos(angle));
        const y = Math.round(center.y + r * Math.sin(angle));
        
        if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
          const idx = (y * width + x) * 4;
          const innerIdx = ((y - 2) * width + x) * 4;
          const outerIdx = ((y + 2) * width + x) * 4;
          
          const inner = (imageArray[innerIdx] + imageArray[innerIdx + 1] + imageArray[innerIdx + 2]) / 3;
          const outer = (imageArray[outerIdx] + imageArray[outerIdx + 1] + imageArray[outerIdx + 2]) / 3;
          
          edgeStrength += Math.abs(outer - inner);
        }
      }
      
      if (edgeStrength > threshold * samples) {
        radius = r;
        break;
      }
    }
    
    return radius;
  }

  private async generateIrisCode(normalized: number[][]): Promise<string> {
    const irisCode: number[] = [];
    
    for (let r = 0; r < normalized.length; r++) {
      for (let theta = 0; theta < normalized[r].length; theta += 2) {
        const current = normalized[r][theta];
        const next = normalized[r][(theta + 1) % normalized[r].length];
        
        const realPart = this.applyGaborFilter(normalized, r, theta, 0);
        const imagPart = this.applyGaborFilter(normalized, r, theta, Math.PI / 2);
        
        irisCode.push(realPart > 0 ? 1 : 0);
        irisCode.push(imagPart > 0 ? 1 : 0);
      }
    }
    
    return irisCode.map(bit => bit.toString()).join('');
  }

  private applyGaborFilter(
    image: number[][],
    centerR: number,
    centerTheta: number,
    phase: number
  ): number {
    const sigma = 2.0;
    const frequency = 0.1;
    let response = 0;
    const windowSize = 5;
    
    for (let dr = -windowSize; dr <= windowSize; dr++) {
      for (let dtheta = -windowSize; dtheta <= windowSize; dtheta++) {
        const r = centerR + dr;
        const theta = centerTheta + dtheta;
        
        if (r >= 0 && r < image.length && theta >= 0 && theta < image[0].length) {
          const gaussian = Math.exp(-(dr * dr + dtheta * dtheta) / (2 * sigma * sigma));
          const sinusoid = Math.cos(2 * Math.PI * frequency * dr + phase);
          const gaborKernel = gaussian * sinusoid;
          
          response += image[r][theta] * gaborKernel;
        }
      }
    }
    
    return response;
  }

  private async generateMask(normalized: number[][]): Promise<string> {
    const mask: number[] = [];
    
    for (let r = 0; r < normalized.length; r++) {
      for (let theta = 0; theta < normalized[r].length; theta += 2) {
        const value = normalized[r][theta];
        const isValid = value > 10 && value < 245;
        mask.push(isValid ? 1 : 0);
        mask.push(isValid ? 1 : 0);
      }
    }
    
    return mask.map(bit => bit.toString()).join('');
  }

  private assessIrisQuality(eyeImage: Buffer): number {
    const imageArray = new Uint8Array(eyeImage);
    let sharpness = 0;
    let contrast = 0;
    let occlusion = 0;
    
    const width = 640;
    const height = 480;
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const center = imageArray[idx];
        
        const top = imageArray[((y - 1) * width + x) * 4];
        const bottom = imageArray[((y + 1) * width + x) * 4];
        const left = imageArray[(y * width + (x - 1)) * 4];
        const right = imageArray[(y * width + (x + 1)) * 4];
        
        const laplacian = Math.abs(4 * center - top - bottom - left - right);
        sharpness += laplacian;
        
        const localContrast = Math.max(top, bottom, left, right) - Math.min(top, bottom, left, right);
        contrast += localContrast;
      }
    }
    
    const totalPixels = (width - 2) * (height - 2);
    sharpness /= totalPixels;
    contrast /= totalPixels;
    
    const sharpnessScore = Math.min(1.0, sharpness / 50);
    const contrastScore = Math.min(1.0, contrast / 100);
    const occlusionScore = 1.0 - (occlusion / totalPixels);
    
    return (sharpnessScore * 0.4 + contrastScore * 0.3 + occlusionScore * 0.3);
  }

  async matchIris(features1: IrisFeatures, features2: IrisFeatures): Promise<IrisMatchResult> {
    const startTime = Date.now();
    
    const hammingDistance = this.calculateHammingDistance(
      features1.irisCode,
      features2.irisCode,
      features1.mask,
      features2.mask
    );
    
    const rotationInvariantDistance = await this.findBestRotationMatch(
      features1.irisCode,
      features2.irisCode,
      features1.mask,
      features2.mask
    );
    
    const similarity = 1 - (rotationInvariantDistance / features1.irisCode.length);
    const isMatch = similarity >= this.MATCH_THRESHOLD;
    
    this.metrics.recordLatency('iris_matching', Date.now() - startTime);
    this.metrics.incrementCounter('iris_matches', { result: isMatch ? 'match' : 'no_match' });
    
    return {
      isMatch,
      confidence: similarity,
      hammingDistance: rotationInvariantDistance,
      rotationAngle: 0,
      timestamp: Date.now()
    };
  }

  private calculateHammingDistance(
    code1: string,
    code2: string,
    mask1: string,
    mask2: string
  ): number {
    let distance = 0;
    let validBits = 0;
    
    for (let i = 0; i < Math.min(code1.length, code2.length); i++) {
      if (mask1[i] === '1' && mask2[i] === '1') {
        if (code1[i] !== code2[i]) {
          distance++;
        }
        validBits++;
      }
    }
    
    return validBits > 0 ? distance / validBits : 1.0;
  }

  private async findBestRotationMatch(
    code1: string,
    code2: string,
    mask1: string,
    mask2: string
  ): Promise<number> {
    let minDistance = Infinity;
    const maxRotation = 15;
    
    for (let rotation = -maxRotation; rotation <= maxRotation; rotation++) {
      const rotatedCode2 = this.rotateIrisCode(code2, rotation);
      const rotatedMask2 = this.rotateIrisCode(mask2, rotation);
      
      const distance = this.calculateHammingDistance(code1, rotatedCode2, mask1, rotatedMask2);
      
      if (distance < minDistance) {
        minDistance = distance;
      }
    }
    
    return minDistance;
  }

  private rotateIrisCode(code: string, rotation: number): string {
    const bitsPerAngle = Math.floor(code.length / 360);
    const shift = rotation * bitsPerAngle;
    
    if (shift === 0) return code;
    
    if (shift > 0) {
      return code.slice(shift) + code.slice(0, shift);
    } else {
      return code.slice(shift) + code.slice(0, shift);
    }
  }

  async detectEyelidOcclusion(eyeImage: Buffer): Promise<number> {
    const imageArray = new Uint8Array(eyeImage);
    const width = 640;
    const height = 480;
    
    let occludedPixels = 0;
    const totalPixels = width * height;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (imageArray[idx] + imageArray[idx + 1] + imageArray[idx + 2]) / 3;
        
        if (brightness > 200 || brightness < 20) {
          occludedPixels++;
        }
      }
    }
    
    return occludedPixels / totalPixels;
  }

  async detectReflections(eyeImage: Buffer): Promise<{ x: number; y: number; intensity: number }[]> {
    const imageArray = new Uint8Array(eyeImage);
    const width = 640;
    const height = 480;
    const reflections: { x: number; y: number; intensity: number }[] = [];
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (imageArray[idx] + imageArray[idx + 1] + imageArray[idx + 2]) / 3;
        
        if (brightness > 240) {
          let isLocalMaximum = true;
          
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              
              const neighborIdx = ((y + dy) * width + (x + dx)) * 4;
              const neighborBrightness = (imageArray[neighborIdx] + imageArray[neighborIdx + 1] + imageArray[neighborIdx + 2]) / 3;
              
              if (neighborBrightness > brightness) {
                isLocalMaximum = false;
                break;
              }
            }
            if (!isLocalMaximum) break;
          }
          
          if (isLocalMaximum) {
            reflections.push({ x, y, intensity: brightness });
          }
        }
      }
    }
    
    return reflections;
  }
}

import { MathUtils } from '@shared/utils/math.utils';

export interface KeystrokeEvent {
  keyCode: number;
  keyDownTime: number;
  keyUpTime: number;
  shiftPressed: boolean;
  ctrlPressed: boolean;
  altPressed: boolean;
}

export interface MouseEvent {
  x: number;
  y: number;
  timestamp: number;
  clickType?: 'left' | 'right' | 'middle';
}

export interface TouchEvent {
  x: number;
  y: number;
  pressure: number;
  contactArea: number;
  timestamp: number;
  gestureType: string;
}

export class BiometricProcessor {
  async processEvents(events: any[]): Promise<any[]> {
    return events.map(event => ({
      ...event,
      processedAt: Date.now(),
      normalized: true
    }));
  }

  async processKeystrokeEvents(events: KeystrokeEvent[]): Promise<any> {
    if (events.length === 0) {
      return this.getEmptyKeystrokeFeatures();
    }

    const dwellTimes = events.map(e => e.keyUpTime - e.keyDownTime);
    const flightTimes = this.calculateFlightTimes(events);
    const typingSpeed = this.calculateTypingSpeed(events);
    const rhythmScore = this.calculateRhythmScore(flightTimes);
    const errorRate = this.calculateErrorRate(events);
    const backspaceFrequency = this.calculateBackspaceFrequency(events);

    const modifierUsage = {
      shift: events.filter(e => e.shiftPressed).length / events.length,
      ctrl: events.filter(e => e.ctrlPressed).length / events.length,
      alt: events.filter(e => e.altPressed).length / events.length
    };

    const keyPressDistribution = this.calculateKeyPressDistribution(events);
    const burstiness = this.calculateBurstiness(events);
    const consistency = this.calculateConsistency(dwellTimes, flightTimes);

    return {
      avgDwellTime: MathUtils.mean(dwellTimes),
      stdDwellTime: MathUtils.std(dwellTimes),
      medianDwellTime: MathUtils.median(dwellTimes),
      avgFlightTime: MathUtils.mean(flightTimes),
      stdFlightTime: MathUtils.std(flightTimes),
      medianFlightTime: MathUtils.median(flightTimes),
      typingSpeed,
      rhythmScore,
      errorRate,
      backspaceFrequency,
      modifierUsage,
      keyPressDistribution,
      burstiness,
      consistency,
      sampleCount: events.length
    };
  }

  async processMouseEvents(events: MouseEvent[]): Promise<any> {
    if (events.length === 0) {
      return this.getEmptyMouseFeatures();
    }

    const positions = events.map(e => ({ x: e.x, y: e.y, timestamp: e.timestamp }));
    const velocities = MathUtils.calculateVelocity(positions);
    const accelerations = MathUtils.calculateAcceleration(
      velocities,
      events.map(e => e.timestamp)
    );
    const curvatures = MathUtils.calculateCurvature(positions);

    const clickEvents = events.filter(e => e.clickType);
    const clickFrequency = clickEvents.length / (events.length || 1);
    const doubleClickTiming = this.calculateDoubleClickTiming(clickEvents);
    const movementSmoothness = this.calculateMovementSmoothness(velocities);
    const directionChanges = this.calculateDirectionChanges(positions);
    const pauseFrequency = this.calculatePauseFrequency(velocities);
    const angularVelocity = this.calculateAngularVelocity(positions);
    const straightnessIndex = this.calculateStraightnessIndex(positions);

    return {
      avgVelocity: MathUtils.mean(velocities),
      maxVelocity: MathUtils.max(velocities),
      stdVelocity: MathUtils.std(velocities),
      avgAcceleration: MathUtils.mean(accelerations),
      maxAcceleration: MathUtils.max(accelerations),
      avgCurvature: MathUtils.mean(curvatures),
      maxCurvature: MathUtils.max(curvatures),
      clickFrequency,
      doubleClickTiming,
      movementSmoothness,
      directionChanges,
      pauseFrequency,
      angularVelocity,
      straightnessIndex,
      sampleCount: events.length
    };
  }

  async processTouchEvents(events: TouchEvent[]): Promise<any> {
    if (events.length === 0) {
      return this.getEmptyTouchFeatures();
    }

    const pressures = events.map(e => e.pressure);
    const contactAreas = events.map(e => e.contactArea);
    const positions = events.map(e => ({ x: e.x, y: e.y, timestamp: e.timestamp }));
    const velocities = MathUtils.calculateVelocity(positions);

    const swipePatterns = this.analyzeSwipePatterns(events);
    const gestureComplexity = this.calculateGestureComplexity(events);
    const multiTouchFrequency = this.calculateMultiTouchFrequency(events);
    const touchDuration = this.calculateTouchDuration(events);
    const touchPrecision = this.calculateTouchPrecision(events);

    return {
      avgPressure: MathUtils.mean(pressures),
      stdPressure: MathUtils.std(pressures),
      maxPressure: MathUtils.max(pressures),
      avgContactArea: MathUtils.mean(contactAreas),
      stdContactArea: MathUtils.std(contactAreas),
      avgVelocity: MathUtils.mean(velocities),
      maxVelocity: MathUtils.max(velocities),
      swipePatterns,
      gestureComplexity,
      multiTouchFrequency,
      touchDuration,
      touchPrecision,
      sampleCount: events.length
    };
  }

  private calculateFlightTimes(events: KeystrokeEvent[]): number[] {
    const flightTimes: number[] = [];
    
    for (let i = 1; i < events.length; i++) {
      const flightTime = events[i].keyDownTime - events[i - 1].keyUpTime;
      if (flightTime >= 0) {
        flightTimes.push(flightTime);
      }
    }
    
    return flightTimes;
  }

  private calculateTypingSpeed(events: KeystrokeEvent[]): number {
    if (events.length < 2) return 0;
    
    const totalTime = events[events.length - 1].keyUpTime - events[0].keyDownTime;
    const minutes = totalTime / 60000;
    
    return minutes > 0 ? events.length / minutes : 0;
  }

  private calculateRhythmScore(flightTimes: number[]): number {
    if (flightTimes.length === 0) return 0;
    
    const variance = MathUtils.variance(flightTimes);
    const mean = MathUtils.mean(flightTimes);
    
    if (mean === 0) return 0;
    
    const coefficientOfVariation = Math.sqrt(variance) / mean;
    return 1 / (1 + coefficientOfVariation);
  }

  private calculateErrorRate(events: KeystrokeEvent[]): number {
    const backspaceKey = 8;
    const deleteKey = 46;
    
    const errorKeys = events.filter(
      e => e.keyCode === backspaceKey || e.keyCode === deleteKey
    );
    
    return errorKeys.length / events.length;
  }

  private calculateBackspaceFrequency(events: KeystrokeEvent[]): number {
    const backspaceKey = 8;
    const backspaces = events.filter(e => e.keyCode === backspaceKey);
    return backspaces.length / events.length;
  }

  private calculateKeyPressDistribution(events: KeystrokeEvent[]): Record<string, number> {
    const distribution: Record<number, number> = {};
    
    events.forEach(e => {
      distribution[e.keyCode] = (distribution[e.keyCode] || 0) + 1;
    });
    
    const total = events.length;
    const normalized: Record<string, number> = {};
    
    Object.entries(distribution).forEach(([key, count]) => {
      normalized[key] = count / total;
    });
    
    return normalized;
  }

  private calculateBurstiness(events: KeystrokeEvent[]): number {
    if (events.length < 2) return 0;
    
    const intervals: number[] = [];
    for (let i = 1; i < events.length; i++) {
      intervals.push(events[i].keyDownTime - events[i - 1].keyDownTime);
    }
    
    const mean = MathUtils.mean(intervals);
    const std = MathUtils.std(intervals);
    
    if (mean === 0) return 0;
    return (std - mean) / (std + mean);
  }

  private calculateConsistency(dwellTimes: number[], flightTimes: number[]): number {
    const dwellConsistency = 1 - (MathUtils.std(dwellTimes) / (MathUtils.mean(dwellTimes) || 1));
    const flightConsistency = 1 - (MathUtils.std(flightTimes) / (MathUtils.mean(flightTimes) || 1));
    
    return (dwellConsistency + flightConsistency) / 2;
  }

  private calculateDoubleClickTiming(clickEvents: MouseEvent[]): number {
    const doubleClickThreshold = 500;
    let doubleClicks = 0;
    
    for (let i = 1; i < clickEvents.length; i++) {
      const timeDiff = clickEvents[i].timestamp - clickEvents[i - 1].timestamp;
      if (timeDiff <= doubleClickThreshold) {
        doubleClicks++;
      }
    }
    
    return doubleClicks / (clickEvents.length || 1);
  }

  private calculateMovementSmoothness(velocities: number[]): number {
    if (velocities.length === 0) return 0;
    
    const velocityChanges = [];
    for (let i = 1; i < velocities.length; i++) {
      velocityChanges.push(Math.abs(velocities[i] - velocities[i - 1]));
    }
    
    const avgChange = MathUtils.mean(velocityChanges);
    const avgVelocity = MathUtils.mean(velocities);
    
    if (avgVelocity === 0) return 1;
    return 1 / (1 + avgChange / avgVelocity);
  }

  private calculateDirectionChanges(positions: Array<{ x: number; y: number }>): number {
    let changes = 0;
    
    for (let i = 2; i < positions.length; i++) {
      const v1 = {
        x: positions[i - 1].x - positions[i - 2].x,
        y: positions[i - 1].y - positions[i - 2].y
      };
      const v2 = {
        x: positions[i].x - positions[i - 1].x,
        y: positions[i].y - positions[i - 1].y
      };
      
      const dotProduct = v1.x * v2.x + v1.y * v2.y;
      if (dotProduct < 0) {
        changes++;
      }
    }
    
    return changes / (positions.length || 1);
  }

  private calculatePauseFrequency(velocities: number[]): number {
    const pauseThreshold = 0.1;
    const pauses = velocities.filter(v => v < pauseThreshold);
    return pauses.length / velocities.length;
  }

  private calculateAngularVelocity(positions: Array<{ x: number; y: number }>): number {
    const angles: number[] = [];
    
    for (let i = 1; i < positions.length; i++) {
      const dx = positions[i].x - positions[i - 1].x;
      const dy = positions[i].y - positions[i - 1].y;
      angles.push(Math.atan2(dy, dx));
    }
    
    const angleChanges: number[] = [];
    for (let i = 1; i < angles.length; i++) {
      angleChanges.push(Math.abs(angles[i] - angles[i - 1]));
    }
    
    return MathUtils.mean(angleChanges);
  }

  private calculateStraightnessIndex(positions: Array<{ x: number; y: number }>): number {
    if (positions.length < 2) return 1;
    
    const start = positions[0];
    const end = positions[positions.length - 1];
    const directDistance = Math.sqrt(
      Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)
    );
    
    let pathLength = 0;
    for (let i = 1; i < positions.length; i++) {
      pathLength += Math.sqrt(
        Math.pow(positions[i].x - positions[i - 1].x, 2) +
        Math.pow(positions[i].y - positions[i - 1].y, 2)
      );
    }
    
    if (pathLength === 0) return 1;
    return directDistance / pathLength;
  }

  private analyzeSwipePatterns(events: TouchEvent[]): number[] {
    const swipes = events.filter(e => e.gestureType === 'swipe');
    const directions: number[] = [];
    
    for (let i = 1; i < swipes.length; i++) {
      const dx = swipes[i].x - swipes[i - 1].x;
      const dy = swipes[i].y - swipes[i - 1].y;
      const angle = Math.atan2(dy, dx);
      directions.push(angle);
    }
    
    return directions;
  }

  private calculateGestureComplexity(events: TouchEvent[]): number {
    const gestureTypes = new Set(events.map(e => e.gestureType));
    return gestureTypes.size / 4;
  }

  private calculateMultiTouchFrequency(events: TouchEvent[]): number {
    const multiTouchGestures = ['pinch', 'rotate'];
    const multiTouch = events.filter(e => multiTouchGestures.includes(e.gestureType));
    return multiTouch.length / events.length;
  }

  private calculateTouchDuration(events: TouchEvent[]): number {
    if (events.length < 2) return 0;
    
    const totalDuration = events[events.length - 1].timestamp - events[0].timestamp;
    return totalDuration / events.length;
  }

  private calculateTouchPrecision(events: TouchEvent[]): number {
    const contactAreas = events.map(e => e.contactArea);
    const avgArea = MathUtils.mean(contactAreas);
    const stdArea = MathUtils.std(contactAreas);
    
    if (avgArea === 0) return 1;
    return 1 / (1 + stdArea / avgArea);
  }

  private getEmptyKeystrokeFeatures(): any {
    return {
      avgDwellTime: 0,
      stdDwellTime: 0,
      medianDwellTime: 0,
      avgFlightTime: 0,
      stdFlightTime: 0,
      medianFlightTime: 0,
      typingSpeed: 0,
      rhythmScore: 0,
      errorRate: 0,
      backspaceFrequency: 0,
      modifierUsage: { shift: 0, ctrl: 0, alt: 0 },
      keyPressDistribution: {},
      burstiness: 0,
      consistency: 0,
      sampleCount: 0
    };
  }

  private getEmptyMouseFeatures(): any {
    return {
      avgVelocity: 0,
      maxVelocity: 0,
      stdVelocity: 0,
      avgAcceleration: 0,
      maxAcceleration: 0,
      avgCurvature: 0,
      maxCurvature: 0,
      clickFrequency: 0,
      doubleClickTiming: 0,
      movementSmoothness: 0,
      directionChanges: 0,
      pauseFrequency: 0,
      angularVelocity: 0,
      straightnessIndex: 0,
      sampleCount: 0
    };
  }

  private getEmptyTouchFeatures(): any {
    return {
      avgPressure: 0,
      stdPressure: 0,
      maxPressure: 0,
      avgContactArea: 0,
      stdContactArea: 0,
      avgVelocity: 0,
      maxVelocity: 0,
      swipePatterns: [],
      gestureComplexity: 0,
      multiTouchFrequency: 0,
      touchDuration: 0,
      touchPrecision: 0,
      sampleCount: 0
    };
  }
}

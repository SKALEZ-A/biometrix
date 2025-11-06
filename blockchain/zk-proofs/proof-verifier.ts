import { groth16 } from 'snarkjs';

export class ProofVerifier {
  private verificationKeys: Map<string, any>;

  constructor() {
    this.verificationKeys = new Map();
  }

  async loadVerificationKey(circuitName: string, keyPath: string): Promise<void> {
    // Load verification key from file
    const vKey = await this.readVerificationKey(keyPath);
    this.verificationKeys.set(circuitName, vKey);
  }

  async verifyProof(circuitName: string, proof: any, publicSignals: any[]): Promise<boolean> {
    const vKey = this.verificationKeys.get(circuitName);
    
    if (!vKey) {
      throw new Error(`Verification key not found for circuit: ${circuitName}`);
    }

    try {
      const isValid = await groth16.verify(vKey, publicSignals, proof);
      return isValid;
    } catch (error) {
      console.error('Proof verification failed:', error);
      return false;
    }
  }

  async verifyBiometricProof(proof: any, publicSignals: any[]): Promise<boolean> {
    return await this.verifyProof('BiometricVerification', proof, publicSignals);
  }

  async verifyFraudDetectionProof(proof: any, publicSignals: any[]): Promise<boolean> {
    return await this.verifyProof('FraudDetection', proof, publicSignals);
  }

  private async readVerificationKey(keyPath: string): Promise<any> {
    // Read verification key from file system
    return {
      protocol: 'groth16',
      curve: 'bn128'
    };
  }

  async batchVerifyProofs(proofs: Array<{ circuitName: string; proof: any; publicSignals: any[] }>): Promise<boolean[]> {
    const results = await Promise.all(
      proofs.map(({ circuitName, proof, publicSignals }) =>
        this.verifyProof(circuitName, proof, publicSignals)
      )
    );

    return results;
  }
}

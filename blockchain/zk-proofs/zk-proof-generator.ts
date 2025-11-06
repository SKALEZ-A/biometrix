import { groth16 } from 'snarkjs';
import { logger } from '@shared/utils/logger';

export class ZKProofGenerator {
  private wasmPath: string;
  private zkeyPath: string;

  constructor(wasmPath: string, zkeyPath: string) {
    this.wasmPath = wasmPath;
    this.zkeyPath = zkeyPath;
  }

  async generateProof(input: any): Promise<{ proof: any; publicSignals: any }> {
    try {
      const { proof, publicSignals } = await groth16.fullProve(
        input,
        this.wasmPath,
        this.zkeyPath
      );

      logger.info('ZK proof generated successfully');
      return { proof, publicSignals };
    } catch (error) {
      logger.error('Failed to generate ZK proof', { error });
      throw error;
    }
  }

  async generateBiometricProof(biometricHash: string, userId: string): Promise<any> {
    const input = {
      biometricHash,
      userId,
      timestamp: Date.now()
    };

    return this.generateProof(input);
  }

  async generateTransactionProof(transactionData: any): Promise<any> {
    const input = {
      amount: transactionData.amount,
      sender: transactionData.sender,
      receiver: transactionData.receiver,
      timestamp: transactionData.timestamp
    };

    return this.generateProof(input);
  }

  exportProof(proof: any): string {
    return JSON.stringify(proof);
  }

  importProof(proofString: string): any {
    return JSON.parse(proofString);
  }
}

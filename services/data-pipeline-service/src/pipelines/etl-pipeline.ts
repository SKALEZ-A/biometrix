import { EventEmitter } from 'events';
import { DataExtractor } from '../extractors/data-extractor';
import { DataTransformer } from '../transformers/data-transformer';
import { DataLoader } from '../loaders/data-loader';
import { DataValidator } from '../validators/data-validator';

export interface PipelineConfig {
  id: string;
  name: string;
  description?: string;
  source: SourceConfig;
  transformations: TransformationConfig[];
  destination: DestinationConfig;
  validation?: ValidationConfig;
  errorHandling?: ErrorHandlingConfig;
  scheduling?: SchedulingConfig;
}

export interface SourceConfig {
  type: 'database' | 'api' | 'file' | 'stream';
  connection: any;
  query?: string;
  path?: string;
  batchSize?: number;
}

export interface TransformationConfig {
  type: string;
  config: any;
  order: number;
}

export interface DestinationConfig {
  type: 'database' | 'file' | 'api' | 'cache';
  connection: any;
  table?: string;
  path?: string;
  mode?: 'append' | 'overwrite' | 'upsert';
}

export interface ValidationConfig {
  schema: any;
  rules: any[];
  onError: 'skip' | 'fail' | 'log';
}

export interface ErrorHandlingConfig {
  retryAttempts: number;
  retryDelay: number;
  deadLetterQueue?: string;
  alertOnFailure: boolean;
}

export interface SchedulingConfig {
  cron?: string;
  interval?: number;
  startTime?: Date;
  endTime?: Date;
}

export interface PipelineResult {
  pipelineId: string;
  status: 'success' | 'partial' | 'failed';
  recordsProcessed: number;
  recordsFailed: number;
  duration: number;
  errors: any[];
  metadata: any;
}

export class ETLPipeline extends EventEmitter {
  private extractor: DataExtractor;
  private transformer: DataTransformer;
  private loader: DataLoader;
  private validator: DataValidator;
  private runningPipelines: Map<string, any>;

  constructor() {
    super();
    this.extractor = new DataExtractor();
    this.transformer = new DataTransformer();
    this.loader = new DataLoader();
    this.validator = new DataValidator();
    this.runningPipelines = new Map();
  }

  async execute(config: PipelineConfig): Promise<PipelineResult> {
    const startTime = Date.now();
    const pipelineId = config.id;

    this.emit('pipeline-started', { pipelineId, config });
    this.runningPipelines.set(pipelineId, { status: 'running', startTime });

    try {
      // Extract data
      this.emit('extraction-started', { pipelineId });
      const extractedData = await this.extractData(config.source);
      this.emit('extraction-completed', { pipelineId, recordCount: extractedData.length });

      // Validate data (if configured)
      let validatedData = extractedData;
      if (config.validation) {
        this.emit('validation-started', { pipelineId });
        validatedData = await this.validateData(extractedData, config.validation);
        this.emit('validation-completed', { pipelineId, validRecords: validatedData.length });
      }

      // Transform data
      this.emit('transformation-started', { pipelineId });
      const transformedData = await this.transformData(validatedData, config.transformations);
      this.emit('transformation-completed', { pipelineId, recordCount: transformedData.length });

      // Load data
      this.emit('loading-started', { pipelineId });
      const loadResult = await this.loadData(transformedData, config.destination);
      this.emit('loading-completed', { pipelineId, result: loadResult });

      const duration = Date.now() - startTime;
      const result: PipelineResult = {
        pipelineId,
        status: 'success',
        recordsProcessed: transformedData.length,
        recordsFailed: extractedData.length - transformedData.length,
        duration,
        errors: [],
        metadata: {
          extractedRecords: extractedData.length,
          validatedRecords: validatedData.length,
          transformedRecords: transformedData.length,
          loadedRecords: loadResult.recordsLoaded
        }
      };

      this.runningPipelines.delete(pipelineId);
      this.emit('pipeline-completed', result);

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      const result: PipelineResult = {
        pipelineId,
        status: 'failed',
        recordsProcessed: 0,
        recordsFailed: 0,
        duration,
        errors: [{ message: error.message, stack: error.stack }],
        metadata: {}
      };

      this.runningPipelines.delete(pipelineId);
      this.emit('pipeline-failed', result);

      if (config.errorHandling?.alertOnFailure) {
        await this.sendFailureAlert(config, error);
      }

      throw error;
    }
  }

  async executeBatch(configs: PipelineConfig[]): Promise<PipelineResult[]> {
    const results: PipelineResult[] = [];

    for (const config of configs) {
      try {
        const result = await this.execute(config);
        results.push(result);
      } catch (error) {
        results.push({
          pipelineId: config.id,
          status: 'failed',
          recordsProcessed: 0,
          recordsFailed: 0,
          duration: 0,
          errors: [{ message: error.message }],
          metadata: {}
        });
      }
    }

    return results;
  }

  async executeParallel(configs: PipelineConfig[]): Promise<PipelineResult[]> {
    const promises = configs.map(config => this.execute(config));
    return Promise.all(promises);
  }

  private async extractData(source: SourceConfig): Promise<any[]> {
    switch (source.type) {
      case 'database':
        return this.extractor.extractFromDatabase(source);
      case 'api':
        return this.extractor.extractFromAPI(source);
      case 'file':
        return this.extractor.extractFromFile(source);
      case 'stream':
        return this.extractor.extractFromStream(source);
      default:
        throw new Error(`Unsupported source type: ${source.type}`);
    }
  }

  private async validateData(data: any[], validation: ValidationConfig): Promise<any[]> {
    const validRecords: any[] = [];
    const invalidRecords: any[] = [];

    for (const record of data) {
      const validationResult = await this.validator.validate(record, validation.schema);

      if (validationResult.valid) {
        validRecords.push(record);
      } else {
        invalidRecords.push({ record, errors: validationResult.errors });

        switch (validation.onError) {
          case 'fail':
            throw new Error(`Validation failed: ${JSON.stringify(validationResult.errors)}`);
          case 'log':
            console.error('Validation error:', validationResult.errors);
            break;
          case 'skip':
            // Skip invalid record
            break;
        }
      }
    }

    return validRecords;
  }

  private async transformData(data: any[], transformations: TransformationConfig[]): Promise<any[]> {
    let transformedData = data;

    // Sort transformations by order
    const sortedTransformations = transformations.sort((a, b) => a.order - b.order);

    for (const transformation of sortedTransformations) {
      transformedData = await this.applyTransformation(transformedData, transformation);
    }

    return transformedData;
  }

  private async applyTransformation(data: any[], transformation: TransformationConfig): Promise<any[]> {
    switch (transformation.type) {
      case 'map':
        return this.transformer.map(data, transformation.config);
      case 'filter':
        return this.transformer.filter(data, transformation.config);
      case 'aggregate':
        return this.transformer.aggregate(data, transformation.config);
      case 'join':
        return this.transformer.join(data, transformation.config);
      case 'pivot':
        return this.transformer.pivot(data, transformation.config);
      case 'normalize':
        return this.transformer.normalize(data, transformation.config);
      case 'enrich':
        return this.transformer.enrich(data, transformation.config);
      case 'deduplicate':
        return this.transformer.deduplicate(data, transformation.config);
      default:
        throw new Error(`Unsupported transformation type: ${transformation.type}`);
    }
  }

  private async loadData(data: any[], destination: DestinationConfig): Promise<any> {
    switch (destination.type) {
      case 'database':
        return this.loader.loadToDatabase(data, destination);
      case 'file':
        return this.loader.loadToFile(data, destination);
      case 'api':
        return this.loader.loadToAPI(data, destination);
      case 'cache':
        return this.loader.loadToCache(data, destination);
      default:
        throw new Error(`Unsupported destination type: ${destination.type}`);
    }
  }

  private async sendFailureAlert(config: PipelineConfig, error: Error): Promise<void> {
    console.error(`Pipeline ${config.id} failed:`, error);
    // In production, send to alerting service
  }

  getPipelineStatus(pipelineId: string): any {
    return this.runningPipelines.get(pipelineId) || { status: 'not_found' };
  }

  getRunningPipelines(): string[] {
    return Array.from(this.runningPipelines.keys());
  }

  async cancelPipeline(pipelineId: string): Promise<void> {
    if (this.runningPipelines.has(pipelineId)) {
      this.runningPipelines.delete(pipelineId);
      this.emit('pipeline-cancelled', { pipelineId });
    }
  }
}

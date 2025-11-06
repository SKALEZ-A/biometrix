import { SourceConfig } from '../pipelines/etl-pipeline';
import * as fs from 'fs';
import * as path from 'path';

export class DataExtractor {
  async extractFromDatabase(source: SourceConfig): Promise<any[]> {
    const { connection, query, batchSize = 1000 } = source;
    
    console.log(`Extracting from database: ${connection.database}`);
    
    // Simulate database extraction
    const mockData = this.generateMockDatabaseData(batchSize);
    
    return mockData;
  }

  async extractFromAPI(source: SourceConfig): Promise<any[]> {
    const { connection, batchSize = 100 } = source;
    
    console.log(`Extracting from API: ${connection.url}`);
    
    try {
      // In production, make actual API calls
      const mockData = this.generateMockAPIData(batchSize);
      return mockData;
    } catch (error) {
      console.error('API extraction error:', error);
      throw error;
    }
  }

  async extractFromFile(source: SourceConfig): Promise<any[]> {
    const { path: filePath, connection } = source;
    
    console.log(`Extracting from file: ${filePath}`);
    
    const fileType = this.getFileType(filePath);
    
    switch (fileType) {
      case 'json':
        return this.extractFromJSON(filePath);
      case 'csv':
        return this.extractFromCSV(filePath);
      case 'xml':
        return this.extractFromXML(filePath);
      case 'parquet':
        return this.extractFromParquet(filePath);
      default:
        throw new Error(`Unsupported file type: ${fileType}`);
    }
  }

  async extractFromStream(source: SourceConfig): Promise<any[]> {
    const { connection, batchSize = 100 } = source;
    
    console.log(`Extracting from stream: ${connection.topic}`);
    
    // Simulate stream extraction
    const mockData = this.generateMockStreamData(batchSize);
    
    return mockData;
  }

  private async extractFromJSON(filePath: string): Promise<any[]> {
    try {
      // Simulate JSON file reading
      return this.generateMockDatabaseData(100);
    } catch (error) {
      console.error('JSON extraction error:', error);
      throw error;
    }
  }

  private async extractFromCSV(filePath: string): Promise<any[]> {
    try {
      // Simulate CSV file reading
      return this.generateMockDatabaseData(100);
    } catch (error) {
      console.error('CSV extraction error:', error);
      throw error;
    }
  }

  private async extractFromXML(filePath: string): Promise<any[]> {
    try {
      // Simulate XML file reading
      return this.generateMockDatabaseData(100);
    } catch (error) {
      console.error('XML extraction error:', error);
      throw error;
    }
  }

  private async extractFromParquet(filePath: string): Promise<any[]> {
    try {
      // Simulate Parquet file reading
      return this.generateMockDatabaseData(100);
    } catch (error) {
      console.error('Parquet extraction error:', error);
      throw error;
    }
  }

  private getFileType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    return ext.substring(1);
  }

  private generateMockDatabaseData(count: number): any[] {
    const data: any[] = [];
    
    for (let i = 0; i < count; i++) {
      data.push({
        id: i + 1,
        transaction_id: `TXN${String(i + 1).padStart(8, '0')}`,
        user_id: `USER${Math.floor(Math.random() * 10000)}`,
        amount: Math.random() * 10000,
        currency: ['USD', 'EUR', 'GBP'][Math.floor(Math.random() * 3)],
        merchant_id: `MERCH${Math.floor(Math.random() * 1000)}`,
        timestamp: new Date(Date.now() - Math.random() * 86400000 * 30).toISOString(),
        status: ['completed', 'pending', 'failed'][Math.floor(Math.random() * 3)],
        payment_method: ['credit_card', 'debit_card', 'bank_transfer'][Math.floor(Math.random() * 3)],
        ip_address: `${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`,
        device_id: `DEV${Math.floor(Math.random() * 100000)}`,
        location: {
          country: ['US', 'UK', 'CA', 'AU'][Math.floor(Math.random() * 4)],
          city: ['New York', 'London', 'Toronto', 'Sydney'][Math.floor(Math.random() * 4)],
          latitude: (Math.random() * 180 - 90).toFixed(6),
          longitude: (Math.random() * 360 - 180).toFixed(6)
        }
      });
    }
    
    return data;
  }

  private generateMockAPIData(count: number): any[] {
    const data: any[] = [];
    
    for (let i = 0; i < count; i++) {
      data.push({
        id: i + 1,
        name: `User ${i + 1}`,
        email: `user${i + 1}@example.com`,
        created_at: new Date(Date.now() - Math.random() * 86400000 * 365).toISOString(),
        updated_at: new Date().toISOString(),
        metadata: {
          source: 'api',
          version: '1.0'
        }
      });
    }
    
    return data;
  }

  private generateMockStreamData(count: number): any[] {
    const data: any[] = [];
    
    for (let i = 0; i < count; i++) {
      data.push({
        event_id: `EVT${String(i + 1).padStart(8, '0')}`,
        event_type: ['click', 'view', 'purchase', 'login'][Math.floor(Math.random() * 4)],
        user_id: `USER${Math.floor(Math.random() * 10000)}`,
        timestamp: Date.now() - Math.random() * 3600000,
        properties: {
          page: `/page${Math.floor(Math.random() * 10)}`,
          referrer: 'https://example.com',
          user_agent: 'Mozilla/5.0'
        }
      });
    }
    
    return data;
  }

  async extractIncremental(source: SourceConfig, lastExtractedTimestamp: Date): Promise<any[]> {
    console.log(`Extracting incremental data since: ${lastExtractedTimestamp}`);
    
    const allData = await this.extractFromDatabase(source);
    
    // Filter data based on timestamp
    return allData.filter(record => {
      const recordTimestamp = new Date(record.timestamp);
      return recordTimestamp > lastExtractedTimestamp;
    });
  }

  async extractWithPagination(source: SourceConfig, page: number, pageSize: number): Promise<any[]> {
    console.log(`Extracting page ${page} with size ${pageSize}`);
    
    const allData = await this.extractFromDatabase(source);
    
    const startIndex = (page - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    
    return allData.slice(startIndex, endIndex);
  }

  async extractWithFilter(source: SourceConfig, filter: any): Promise<any[]> {
    console.log(`Extracting with filter:`, filter);
    
    const allData = await this.extractFromDatabase(source);
    
    return allData.filter(record => {
      for (const key in filter) {
        if (record[key] !== filter[key]) {
          return false;
        }
      }
      return true;
    });
  }

  async extractSchema(source: SourceConfig): Promise<any> {
    console.log(`Extracting schema from source`);
    
    const sampleData = await this.extractFromDatabase({ ...source, batchSize: 1 });
    
    if (sampleData.length === 0) {
      return {};
    }
    
    const schema: any = {};
    const sample = sampleData[0];
    
    for (const key in sample) {
      schema[key] = {
        type: typeof sample[key],
        nullable: sample[key] === null,
        example: sample[key]
      };
    }
    
    return schema;
  }

  async validateConnection(source: SourceConfig): Promise<boolean> {
    try {
      console.log(`Validating connection to source`);
      
      // Simulate connection validation
      await new Promise(resolve => setTimeout(resolve, 100));
      
      return true;
    } catch (error) {
      console.error('Connection validation failed:', error);
      return false;
    }
  }

  async getRecordCount(source: SourceConfig): Promise<number> {
    console.log(`Getting record count from source`);
    
    const data = await this.extractFromDatabase(source);
    return data.length;
  }
}

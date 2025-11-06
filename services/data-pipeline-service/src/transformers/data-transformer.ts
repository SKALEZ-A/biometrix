export class DataTransformer {
  async transform(data: any[], rules: any): Promise<any[]> {
    let transformedData = data;

    for (const rule of rules) {
      transformedData = await this.applyRule(transformedData, rule);
    }

    return transformedData;
  }

  async map(data: any[], config: any): Promise<any[]> {
    const { mapping } = config;

    return data.map(record => {
      const mapped: any = {};

      for (const [targetField, sourceField] of Object.entries(mapping)) {
        if (typeof sourceField === 'string') {
          mapped[targetField] = this.getNestedValue(record, sourceField);
        } else if (typeof sourceField === 'function') {
          mapped[targetField] = sourceField(record);
        }
      }

      return mapped;
    });
  }

  async filter(data: any[], config: any): Promise<any[]> {
    const { conditions } = config;

    return data.filter(record => {
      return this.evaluateConditions(record, conditions);
    });
  }

  async aggregate(data: any[], config: any): Promise<any[]> {
    const { groupBy, aggregations } = config;

    const groups = new Map<string, any[]>();

    // Group data
    for (const record of data) {
      const groupKey = groupBy.map((field: string) => record[field]).join('|');

      if (!groups.has(groupKey)) {
        groups.set(groupKey, []);
      }

      groups.get(groupKey)!.push(record);
    }

    // Aggregate each group
    const results: any[] = [];

    for (const [groupKey, records] of groups.entries()) {
      const aggregated: any = {};

      // Add group by fields
      groupBy.forEach((field: string, index: number) => {
        aggregated[field] = groupKey.split('|')[index];
      });

      // Apply aggregations
      for (const [field, aggFunc] of Object.entries(aggregations)) {
        aggregated[field] = this.applyAggregation(records, field, aggFunc as string);
      }

      results.push(aggregated);
    }

    return results;
  }

  async join(data: any[], config: any): Promise<any[]> {
    const { rightData, leftKey, rightKey, joinType = 'inner' } = config;

    const results: any[] = [];

    for (const leftRecord of data) {
      const leftKeyValue = this.getNestedValue(leftRecord, leftKey);
      let matched = false;

      for (const rightRecord of rightData) {
        const rightKeyValue = this.getNestedValue(rightRecord, rightKey);

        if (leftKeyValue === rightKeyValue) {
          results.push({ ...leftRecord, ...rightRecord });
          matched = true;
        }
      }

      if (!matched && (joinType === 'left' || joinType === 'outer')) {
        results.push(leftRecord);
      }
    }

    if (joinType === 'right' || joinType === 'outer') {
      for (const rightRecord of rightData) {
        const rightKeyValue = this.getNestedValue(rightRecord, rightKey);
        const hasMatch = data.some(leftRecord => 
          this.getNestedValue(leftRecord, leftKey) === rightKeyValue
        );

        if (!hasMatch) {
          results.push(rightRecord);
        }
      }
    }

    return results;
  }

  async pivot(data: any[], config: any): Promise<any[]> {
    const { index, columns, values, aggFunc = 'sum' } = config;

    const pivotMap = new Map<string, Map<string, number>>();

    for (const record of data) {
      const indexValue = record[index];
      const columnValue = record[columns];
      const value = record[values];

      if (!pivotMap.has(indexValue)) {
        pivotMap.set(indexValue, new Map());
      }

      const row = pivotMap.get(indexValue)!;
      const currentValue = row.get(columnValue) || 0;

      switch (aggFunc) {
        case 'sum':
          row.set(columnValue, currentValue + value);
          break;
        case 'count':
          row.set(columnValue, currentValue + 1);
          break;
        case 'avg':
          // Store sum and count separately for average
          row.set(columnValue, currentValue + value);
          break;
        case 'max':
          row.set(columnValue, Math.max(currentValue, value));
          break;
        case 'min':
          row.set(columnValue, currentValue === 0 ? value : Math.min(currentValue, value));
          break;
      }
    }

    const results: any[] = [];

    for (const [indexValue, row] of pivotMap.entries()) {
      const result: any = { [index]: indexValue };

      for (const [columnValue, value] of row.entries()) {
        result[columnValue] = value;
      }

      results.push(result);
    }

    return results;
  }

  async normalize(data: any[], config: any): Promise<any[]> {
    const { fields, method = 'min-max' } = config;

    const stats = this.calculateStatistics(data, fields);

    return data.map(record => {
      const normalized = { ...record };

      for (const field of fields) {
        const value = record[field];

        if (typeof value !== 'number') continue;

        switch (method) {
          case 'min-max':
            normalized[field] = (value - stats[field].min) / (stats[field].max - stats[field].min);
            break;
          case 'z-score':
            normalized[field] = (value - stats[field].mean) / stats[field].stdDev;
            break;
          case 'decimal-scaling':
            const maxAbs = Math.max(Math.abs(stats[field].min), Math.abs(stats[field].max));
            const j = Math.ceil(Math.log10(maxAbs));
            normalized[field] = value / Math.pow(10, j);
            break;
        }
      }

      return normalized;
    });
  }

  async enrich(data: any[], config: any): Promise<any[]> {
    const { enrichmentSource, enrichmentKey, targetFields } = config;

    return data.map(record => {
      const enriched = { ...record };
      const keyValue = record[enrichmentKey];

      // Simulate enrichment lookup
      const enrichmentData = this.lookupEnrichmentData(keyValue, enrichmentSource);

      if (enrichmentData) {
        for (const field of targetFields) {
          enriched[field] = enrichmentData[field];
        }
      }

      return enriched;
    });
  }

  async deduplicate(data: any[], config: any): Promise<any[]> {
    const { keys, strategy = 'first' } = config;

    const seen = new Map<string, any>();

    for (const record of data) {
      const key = keys.map((k: string) => record[k]).join('|');

      if (!seen.has(key)) {
        seen.set(key, record);
      } else if (strategy === 'last') {
        seen.set(key, record);
      } else if (strategy === 'merge') {
        const existing = seen.get(key);
        seen.set(key, { ...existing, ...record });
      }
    }

    return Array.from(seen.values());
  }

  async sort(data: any[], config: any): Promise<any[]> {
    const { fields, orders } = config;

    return data.sort((a, b) => {
      for (let i = 0; i < fields.length; i++) {
        const field = fields[i];
        const order = orders[i] || 'asc';

        const aValue = this.getNestedValue(a, field);
        const bValue = this.getNestedValue(b, field);

        if (aValue < bValue) return order === 'asc' ? -1 : 1;
        if (aValue > bValue) return order === 'asc' ? 1 : -1;
      }

      return 0;
    });
  }

  async split(data: any[], config: any): Promise<any[][]> {
    const { field, delimiter } = config;

    const groups = new Map<string, any[]>();

    for (const record of data) {
      const value = record[field];
      const parts = value.split(delimiter);

      for (const part of parts) {
        if (!groups.has(part)) {
          groups.set(part, []);
        }

        groups.get(part)!.push(record);
      }
    }

    return Array.from(groups.values());
  }

  async flatten(data: any[], config: any): Promise<any[]> {
    const { nestedField, prefix = '' } = config;

    return data.map(record => {
      const flattened = { ...record };
      const nested = record[nestedField];

      if (nested && typeof nested === 'object') {
        delete flattened[nestedField];

        for (const [key, value] of Object.entries(nested)) {
          flattened[`${prefix}${key}`] = value;
        }
      }

      return flattened;
    });
  }

  async unflatten(data: any[], config: any): Promise<any[]> {
    const { targetField, prefix } = config;

    return data.map(record => {
      const unflattened = { ...record };
      const nested: any = {};

      for (const [key, value] of Object.entries(record)) {
        if (key.startsWith(prefix)) {
          const nestedKey = key.substring(prefix.length);
          nested[nestedKey] = value;
          delete unflattened[key];
        }
      }

      unflattened[targetField] = nested;

      return unflattened;
    });
  }

  async cast(data: any[], config: any): Promise<any[]> {
    const { fields, types } = config;

    return data.map(record => {
      const casted = { ...record };

      for (let i = 0; i < fields.length; i++) {
        const field = fields[i];
        const type = types[i];
        const value = record[field];

        switch (type) {
          case 'string':
            casted[field] = String(value);
            break;
          case 'number':
            casted[field] = Number(value);
            break;
          case 'boolean':
            casted[field] = Boolean(value);
            break;
          case 'date':
            casted[field] = new Date(value);
            break;
          case 'json':
            casted[field] = JSON.parse(value);
            break;
        }
      }

      return casted;
    });
  }

  private async applyRule(data: any[], rule: any): Promise<any[]> {
    switch (rule.type) {
      case 'rename':
        return this.renameFields(data, rule.config);
      case 'remove':
        return this.removeFields(data, rule.config);
      case 'add':
        return this.addFields(data, rule.config);
      case 'replace':
        return this.replaceValues(data, rule.config);
      default:
        return data;
    }
  }

  private renameFields(data: any[], config: any): any[] {
    const { mapping } = config;

    return data.map(record => {
      const renamed: any = {};

      for (const [oldName, newName] of Object.entries(mapping)) {
        if (oldName in record) {
          renamed[newName as string] = record[oldName];
        }
      }

      // Keep fields not in mapping
      for (const [key, value] of Object.entries(record)) {
        if (!(key in mapping)) {
          renamed[key] = value;
        }
      }

      return renamed;
    });
  }

  private removeFields(data: any[], config: any): any[] {
    const { fields } = config;

    return data.map(record => {
      const filtered: any = {};

      for (const [key, value] of Object.entries(record)) {
        if (!fields.includes(key)) {
          filtered[key] = value;
        }
      }

      return filtered;
    });
  }

  private addFields(data: any[], config: any): any[] {
    const { fields } = config;

    return data.map(record => {
      const extended = { ...record };

      for (const [field, value] of Object.entries(fields)) {
        if (typeof value === 'function') {
          extended[field] = value(record);
        } else {
          extended[field] = value;
        }
      }

      return extended;
    });
  }

  private replaceValues(data: any[], config: any): any[] {
    const { field, replacements } = config;

    return data.map(record => {
      const replaced = { ...record };

      if (field in record) {
        const value = record[field];
        replaced[field] = replacements[value] !== undefined ? replacements[value] : value;
      }

      return replaced;
    });
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  private evaluateConditions(record: any, conditions: any): boolean {
    for (const [field, condition] of Object.entries(conditions)) {
      const value = this.getNestedValue(record, field);

      if (typeof condition === 'object') {
        for (const [operator, operand] of Object.entries(condition)) {
          if (!this.evaluateOperator(value, operator, operand)) {
            return false;
          }
        }
      } else {
        if (value !== condition) {
          return false;
        }
      }
    }

    return true;
  }

  private evaluateOperator(value: any, operator: string, operand: any): boolean {
    switch (operator) {
      case '$eq':
        return value === operand;
      case '$ne':
        return value !== operand;
      case '$gt':
        return value > operand;
      case '$gte':
        return value >= operand;
      case '$lt':
        return value < operand;
      case '$lte':
        return value <= operand;
      case '$in':
        return operand.includes(value);
      case '$nin':
        return !operand.includes(value);
      case '$regex':
        return new RegExp(operand).test(value);
      default:
        return false;
    }
  }

  private applyAggregation(records: any[], field: string, aggFunc: string): any {
    const values = records.map(r => r[field]).filter(v => v !== null && v !== undefined);

    switch (aggFunc) {
      case 'sum':
        return values.reduce((a, b) => a + b, 0);
      case 'avg':
        return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
      case 'count':
        return values.length;
      case 'min':
        return Math.min(...values);
      case 'max':
        return Math.max(...values);
      case 'first':
        return values[0];
      case 'last':
        return values[values.length - 1];
      default:
        return null;
    }
  }

  private calculateStatistics(data: any[], fields: string[]): any {
    const stats: any = {};

    for (const field of fields) {
      const values = data.map(r => r[field]).filter(v => typeof v === 'number');

      if (values.length === 0) continue;

      const sum = values.reduce((a, b) => a + b, 0);
      const mean = sum / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;

      stats[field] = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean,
        stdDev: Math.sqrt(variance),
        sum,
        count: values.length
      };
    }

    return stats;
  }

  private lookupEnrichmentData(key: string, source: string): any {
    // Simulate enrichment lookup
    return {
      enriched_field_1: `enriched_${key}`,
      enriched_field_2: Math.random() * 100
    };
  }
}

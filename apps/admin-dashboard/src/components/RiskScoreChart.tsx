import React, { useEffect, useState } from 'react';

interface RiskData {
  timestamp: string;
  averageScore: number;
  highRiskCount: number;
}

export const RiskScoreChart: React.FC = () => {
  const [data, setData] = useState<RiskData[]>([]);

  useEffect(() => {
    fetchRiskData();
    const interval = setInterval(fetchRiskData, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchRiskData = async () => {
    try {
      const response = await fetch('/api/analytics/risk-trends?hours=24');
      const result = await response.json();
      setData(result.data);
    } catch (error) {
      console.error('Failed to fetch risk data:', error);
    }
  };

  const maxScore = Math.max(...data.map(d => d.averageScore), 100);

  return (
    <div className="risk-score-chart">
      <h2>Risk Score Trends (24h)</h2>
      <div className="chart-container">
        <svg width="100%" height="300" viewBox="0 0 800 300">
          <g>
            {data.map((point, index) => {
              const x = (index / (data.length - 1)) * 750 + 25;
              const y = 275 - (point.averageScore / maxScore) * 250;
              return (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#3b82f6"
                />
              );
            })}
            <polyline
              points={data.map((point, index) => {
                const x = (index / (data.length - 1)) * 750 + 25;
                const y = 275 - (point.averageScore / maxScore) * 250;
                return `${x},${y}`;
              }).join(' ')}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
            />
          </g>
        </svg>
      </div>
    </div>
  );
};

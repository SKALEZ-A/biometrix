import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import BiometricCapture from './components/BiometricCapture';
import Dashboard from './pages/Dashboard';
import Alerts from './pages/Alerts';
import Login from './pages/Login';
import { biometricService, FraudResponse } from './services/biometricService';
import { FaceEmbedding } from './components/BiometricCapture';
import './assets/styles/global.css';  // Global styles

const App: React.FC = () => {
  const [userId, setUserId] = useState<string | null>(localStorage.getItem('userId'));
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [alerts, setAlerts] = useState<FraudResponse[]>([]);
  const [fraudCount, setFraudCount] = useState(0);

  useEffect(() => {
    if (userId) {
      setIsAuthenticated(true);
      loadAlerts();
    }
  }, [userId]);

  const loadAlerts = async () => {
    if (!userId) return;
    const userAlerts = await biometricService.getAlerts(userId);
    setAlerts(userAlerts);
    setFraudCount(userAlerts.filter(a => a.fraud_detected).length);
  };

  const handleLogin = (id: string) => {
    localStorage.setItem('userId', id);
    setUserId(id);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('userId');
    setUserId(null);
    setIsAuthenticated(false);
    setAlerts([]);
  };

  const handleCapture = async (embedding: FaceEmbedding) => {
    if (!userId) return;
    try {
      const result = await biometricService.detectFraud(embedding, userId);
      if (result.fraud_detected) {
        alert(`Fraud detected! Score: ${result.score}`);
        setFraudCount(prev => prev + 1);
      } else {
        alert('Verification successful.');
      }
      loadAlerts();  // Refresh alerts
    } catch (error) {
      console.error('Capture failed:', error);
    }
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <Link to="/" className="nav-link">Dashboard</Link>
          <Link to="/capture" className="nav-link">Biometric Capture</Link>
          <Link to="/alerts" className="nav-link">Alerts ({fraudCount})</Link>
          <button onClick={handleLogout} className="logout-btn">Logout</button>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard alerts={alerts} />} />
            <Route path="/capture" element={<BiometricCapture onCapture={handleCapture} userId={userId!} />} />
            <Route path="/alerts" element={<Alerts alerts={alerts} />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;

// Additional utility hooks and context for state management
export const useAuth = () => {
  const userId = localStorage.getItem('userId');
  return { userId, isAuthenticated: !!userId };
};

export const FraudContext = React.createContext({ fraudCount: 0, updateFraudCount: (count: number) => {} });

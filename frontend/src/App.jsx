import { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import Analyzer from './pages/Analyzer';
import Dashboard from './pages/Dashboard';
import Investigation from './pages/Investigation';
import Performance from './pages/Performance';
import { api } from './lib/api';

export default function App() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const poll = () => api.health().then(setHealth).catch(() => setHealth({ model_loaded: false }));
    poll();
    const timer = setInterval(poll, 30000);
    return () => clearInterval(timer);
  }, []);

  return (
    <Layout health={health}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analyze" element={<Analyzer />} />
        <Route path="/investigate" element={<Investigation />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

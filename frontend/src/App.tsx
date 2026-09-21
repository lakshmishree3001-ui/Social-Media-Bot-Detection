import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/common/Header';
import { Footer } from './components/common/Footer';
import { AnalystConsole } from './pages/AnalystConsole';
import { CommunitiesView } from './pages/CommunitiesView';
import { CoordinationView } from './pages/CoordinationView';
import { TemporalView } from './pages/TemporalView';
import { ModelRegistryView } from './pages/ModelRegistryView';
import { LandingPage } from './pages/LandingPage';
import { ResponsibleUsePage } from './pages/ResponsibleUsePage';
import { DatasetPage } from './pages/DatasetPage';
import { DatasetManagementPage } from './pages/DatasetManagementPage';
import { TermsPage } from './pages/TermsPage';
import { PrivacyPage } from './pages/PrivacyPage';
import './styles/tokens.css';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Header />
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/app" element={<AnalystConsole />} />
            <Route path="/app/communities" element={<CommunitiesView />} />
            <Route path="/app/coordination" element={<CoordinationView />} />
            <Route path="/app/temporal" element={<TemporalView />} />
            <Route path="/app/models" element={<ModelRegistryView />} />
            <Route path="/app/datasets" element={<DatasetManagementPage />} />
            <Route path="/responsible-use" element={<ResponsibleUsePage />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/terms" element={<TermsPage />} />
            <Route path="/privacy" element={<PrivacyPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
};

export default App;


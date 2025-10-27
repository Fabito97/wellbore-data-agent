// src/features/analysis/analysisSlice.ts
import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export interface NodalAnalysisResult {
  id: string;
  wellName: string;
  productionRate: number;
  optimalPressure: number;
  iprCurve: Array<{ pressure: number; rate: number }>;
  vlpCurve: Array<{ pressure: number; rate: number }>;
  timestamp: number;
}

interface AnalysisState {
  results: NodalAnalysisResult[];
  currentAnalysisId: string | null;
  isCalculating: boolean;
}

const initialState: AnalysisState = {
  results: [],
  currentAnalysisId: null,
  isCalculating: false,
};

const analysisSlice = createSlice({
  name: 'analysis',
  initialState,
  reducers: {
    addAnalysisResult: (state, action: PayloadAction<NodalAnalysisResult>) => {
      state.results.push(action.payload);
      state.currentAnalysisId = action.payload.id;
    },
    
    setCurrentAnalysis: (state, action: PayloadAction<string | null>) => {
      state.currentAnalysisId = action.payload;
    },
    
    setIsCalculating: (state, action: PayloadAction<boolean>) => {
      state.isCalculating = action.payload;
    },
    
    clearResults: (state) => {
      state.results = [];
      state.currentAnalysisId = null;
    },
  },
});

export const {
  addAnalysisResult,
  setCurrentAnalysis,
  setIsCalculating,
  clearResults,
} = analysisSlice.actions;

export default analysisSlice.reducer;
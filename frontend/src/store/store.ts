// src/app/store.ts
import { configureStore } from "@reduxjs/toolkit";
import { setupListeners } from "@reduxjs/toolkit/query";
import chatReducer from "./slice/chatSlice";
import documentsReducer from "./slice/documentsSlice";
import analysisReducer from "./slice/analysisSlice";
// import validationReducer from '../features/validation/validationSlice';
import { chatApi } from "./api/chatApi";
import { documentsApi } from "./api/documentsApi";
// import { analysisApi } from '../features/analysis/analysisApi';

export const store = configureStore({
  reducer: {
    chat: chatReducer,
    documents: documentsReducer,
    analysis: analysisReducer,
    // validation: validationReducer,
    // RTK Query reducers
    [chatApi.reducerPath]: chatApi.reducer,
    [documentsApi.reducerPath]: documentsApi.reducer,
    // [analysisApi.reducerPath]: analysisApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware()
      .concat(chatApi.middleware)
      .concat(documentsApi.middleware),
  // .concat(analysisApi.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// import React from 'react'

import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

const RootLayout = () => {
  return (
    <div className="flex flex-col h-screen text-gray-800 dark:text-gray-100 dark:bg-primary-backgroun font-sans text-gray-200">
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 flex flex-col h-full text-gray-800 dark:text-gray-100">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default RootLayout;

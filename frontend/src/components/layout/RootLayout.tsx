// import React from 'react'

import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import Header from "./Header";

const RootLayout = () => {
  return (
    <div className="flex flex-col h-screen text-gray-800 dark:text-gray-100 dark:bg-primary-backgroun font-sans text-gray-200 overflow-hidde">
      <div className="flex flex-1 h-full overflow-hidde">
        <Sidebar />
        <main className="flex-1 flex flex-col h-full text-gray-800 dark:text-gray-100">
          <div className="relative h-full overflow-hidden">
            <Header />

            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default RootLayout;

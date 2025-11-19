
import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex space-x-1.5">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
    </div>
  );
};

export default LoadingSpinner;

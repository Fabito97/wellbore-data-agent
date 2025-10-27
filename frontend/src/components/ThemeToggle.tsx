import { useTheme } from '../context/ThemeContext';

export const ThemeToggle = () => {
  const { theme, setTheme } = useTheme();
  const isDarkMode = theme === "dark";

  const toggleTheme = () => {
    setTheme(isDarkMode ? "light" : "dark");
  };

  return (
    <div className="flex gap-2 hover:opacity-90 text-xs">
      {isDarkMode 
        ?<button className='cursor-pointer py-1 border border-gray-500! px-2 rounded-full dark:border-gray-400 ' onClick={() => toggleTheme()}>☀️ Light</button>
        : <button className='cursor-pointer py-1 px-2 border border-gray-500! rounded-full dark:border-gray-400 ' onClick={() => toggleTheme()}>🌙 Dark</button>
      }      
    </div>
  );
};
// ThemeContext.tsx
import { createContext, useState, useEffect, useContext, type ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextProps {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextProps | undefined>(undefined);

const getSystemTheme = (): 'light' | 'dark' =>
  window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

const getInitialTheme = (): Theme => {
  return (localStorage.getItem('theme') as Theme) || 'system';
};

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [activeTheme, setActiveTheme] = useState<'light' | 'dark'>(() =>
    theme === 'system' ? getSystemTheme() : theme
  );

  useEffect(() => {
    localStorage.setItem('theme', theme);

    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const handleSystemChange = (e: MediaQueryListEvent) =>
        setActiveTheme(e.matches ? 'dark' : 'light');

      mediaQuery.addEventListener('change', handleSystemChange);
      setActiveTheme(getSystemTheme());

      return () => mediaQuery.removeEventListener('change', handleSystemChange);
    } else {
      setActiveTheme(theme);
    }
  }, [theme]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', activeTheme === 'dark');
  }, [activeTheme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within a ThemeProvider');
  return context;
};
const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export function apiUrl(path: string): string {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL;
  return `${baseUrl.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;
}

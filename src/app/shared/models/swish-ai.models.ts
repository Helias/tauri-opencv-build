export interface Stats {
  shots: number;
  baskets: number;
  accuracy: number;
}

export interface Thresholds {
  [key: number]: number;
  0: number; // Ball
  1: number; // Ball in Basket
  2: number; // Player
  3: number; // Basket
  4: number; // Player Shooting
}

export interface StatusResponse {
  status:
    | 'idle'
    | 'uploading'
    | 'processing'
    | 'completed'
    | 'error'
    | 'stopped';
  percentage: number;
  stats: Stats;
  message?: string;
}

export interface UploadResponse {
  file_id: string;
}

export type ProcessingMode = 'full_tracking' | 'stats_effects' | 'stats_only';

export const DEFAULT_THRESHOLDS: Thresholds = {
  0: 0.6, // Ball
  1: 0.25, // Ball in Basket
  2: 0.7, // Player
  3: 0.7, // Basket
  4: 0.77, // Player Shooting
};

export const LABELS: Record<number, string> = {
  0: 'Ball (Orange)',
  1: 'Ball in Basket (Gold)',
  2: 'Player (Green)',
  3: 'Basket (Red)',
  4: 'Player Shooting (Blue)',
};

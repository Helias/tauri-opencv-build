import { Injectable } from '@angular/core';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { save } from '@tauri-apps/plugin-dialog';
import { from, Observable, Subject } from 'rxjs';

export interface ProcessingProgress {
  progress: number;
  current_frame: number;
  total_frames: number;
  status: string;
  shots_attempted?: number;
  baskets_made?: number;
  accuracy?: number;
  processing_time?: number;
  output_path?: string;
}

@Injectable({
  providedIn: 'root',
})
export class TauriService {
  private progressSubject = new Subject<ProcessingProgress>();
  public progress$ = this.progressSubject.asObservable();

  get isTauri(): boolean {
    return !!(window && window.__TAURI__);
  }

  constructor() {
    if (this.isTauri) {
      this.setupProgressListener();
    }
  }

  private async setupProgressListener() {
    await listen<ProcessingProgress>('processing-progress', (event) => {
      this.progressSubject.next(event.payload);
    });
  }

  async selectSaveLocation(
    defaultName: string = 'output.mp4',
  ): Promise<string | null> {
    const filePath = await save({
      defaultPath: defaultName,
      filters: [
        {
          name: 'Video',
          extensions: ['mp4'],
        },
      ],
    });

    return filePath;
  }

  async processFile(inputPath: string): Promise<void> {
    await invoke<string>('process_file', {
      filePath: inputPath,
    });
  }

  getProcessingProgress(): Observable<ProcessingProgress> {
    return from(invoke<ProcessingProgress>('get_processing_progress'));
  }

  async copyFile(sourcePath: string, destPath: string): Promise<void> {
    await invoke<void>('copy_file', {
      sourcePath,
      destPath,
    });
  }
}

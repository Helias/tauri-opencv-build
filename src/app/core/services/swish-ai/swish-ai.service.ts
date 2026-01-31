import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Observable } from 'rxjs';
import {
  ProcessingMode,
  StatusResponse,
  Thresholds,
  UploadResponse,
} from '../../../shared/models/swish-ai.models';
import { TauriService } from '../tauri/tauri.service';

@Injectable({
  providedIn: 'root',
})
export class SwishAiService {
  private readonly API_URL = 'http://localhost:8000';

  private readonly http = inject(HttpClient);
  private readonly tauriService = inject(TauriService);

  async selectVideoFile(): Promise<string | null> {
    if (!this.tauriService.isTauri) {
      return null;
    }

    const selected = await open({
      multiple: false,
      filters: [
        {
          name: 'Video',
          extensions: ['mp4', 'mov', 'avi'],
        },
      ],
    });

    return selected as string | null;
  }

  uploadVideo(file: File): Observable<UploadResponse> {
    console.log('file', file);

    if (this.tauriService.isTauri) {
      // In Tauri mode, this method shouldn't be called since we handle files differently
      console.warn(
        'uploadVideo called in Tauri mode - use processVideoInTauri instead',
      );
    }

    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.API_URL}/upload`, formData);
  }

  async processVideoInTauri(inputPath: string): Promise<void> {
    // Start processing without asking for output location
    // The output will be saved to a temporary location
    await this.tauriService.processFile(inputPath);
  }

  getProcessingProgress(): Observable<any> {
    return this.tauriService.getProcessingProgress();
  }

  startProcessing(
    fileId: string,
    testMode: boolean,
    processingMode: ProcessingMode,
    thresholds: Thresholds,
  ): Observable<any> {
    const thresholdParam = encodeURIComponent(JSON.stringify(thresholds));
    return this.http.post(
      `${this.API_URL}/process/${fileId}?test_mode=${testMode}&mode=${processingMode}&thresholds=${thresholdParam}`,
      {},
    );
  }

  getStatus(fileId: string): Observable<StatusResponse> {
    return this.http.get<StatusResponse>(`${this.API_URL}/status/${fileId}`);
  }

  stopProcessing(fileId: string): Observable<any> {
    return this.http.post(`${this.API_URL}/stop/${fileId}`, {});
  }

  getDownloadUrl(fileId: string): string {
    return `${this.API_URL}/download/${fileId}`;
  }
}

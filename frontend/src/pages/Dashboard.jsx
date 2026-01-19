import React, { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Upload,
  Loader2,
  AlertCircle,
  CheckCircle,
  Leaf,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import PredictionResult from "@/components/PredictionResult";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = React.useRef(null);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please select a valid image file (JPEG, PNG, GIF, or WebP)");
      return;
    }

    const maxSize = 25 * 1024 * 1024;
    if (selectedFile.size > maxSize) {
      setError("File size exceeds 25MB limit");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setPrediction(null);
    setSuccess(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("drag-over");
  };

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove("drag-over");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-over");
    const droppedFile = e.dataTransfer?.files?.[0];
    if (droppedFile) {
      const fakeEvent = { target: { files: [droppedFile] } };
      handleFileSelect(fakeEvent);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API}/predictions/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const result = await response.json();
      setSuccess("Prediction completed successfully!");
      setPrediction(result);
    } catch (err) {
      setError(err.message || "Failed to upload and process image");
    } finally {
      setLoading(false);
    }
  };

  const handleNewScan = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setSuccess(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="min-h-screen bg-[#F4F5F0] py-8 px-4"
    >
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              data-testid="back-home-btn"
              variant="outline"
              onClick={() => navigate("/")}
              className="border-[#1A3C34] text-[#1A3C34] hover:bg-[#1A3C34] hover:text-white"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div className="flex items-center gap-2">
              <Leaf className="w-8 h-8 text-[#1A3C34]" />
              <h1
                className="text-3xl font-bold text-[#1A3C34]"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Disease Detection
              </h1>
            </div>
          </div>
          <Button
            data-testid="view-history-btn"
            variant="outline"
            onClick={() => navigate("/history")}
            className="border-[#1A3C34] text-[#1A3C34] hover:bg-[#1A3C34] hover:text-white"
          >
            View History
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card className="bg-white border border-[#1A3C34]/10 rounded-sm result-card">
              <CardHeader>
                <CardTitle
                  className="text-[#1A3C34]"
                  style={{ fontFamily: "'Playfair Display', serif" }}
                >
                  Upload Plant Image
                </CardTitle>
                <CardDescription className="text-[#5C6B66]">
                  Select an image to detect plant diseases
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div
                  data-testid="upload-zone"
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className="upload-zone border-2 border-dashed border-[#1A3C34]/30 bg-[#1A3C34]/5 rounded-lg p-12 text-center cursor-pointer transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="mx-auto h-12 w-12 text-[#1A3C34]/50 mb-4" />
                  <p className="text-sm font-medium text-[#1A3C34] mb-2">
                    Drag and drop your image here
                  </p>
                  <p className="text-xs text-[#5C6B66] mb-4">
                    or click to browse
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    data-testid="file-input"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    className="bg-transparent border border-[#1A3C34] text-[#1A3C34] hover:bg-[#1A3C34] hover:text-white rounded-none px-6 py-2 text-xs uppercase tracking-wider"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    Browse Files
                  </Button>
                </div>

                {preview && (
                  <div className="relative">
                    <img
                      data-testid="preview-image"
                      src={preview}
                      alt="Preview"
                      className="w-full rounded-lg max-h-64 object-cover border-2 border-[#1A3C34]/10"
                    />
                    <p
                      className="text-xs text-[#5C6B66] mt-2"
                      style={{ fontFamily: "'JetBrains Mono', monospace" }}
                    >
                      {file?.name} ({(file?.size / 1024).toFixed(2)} KB)
                    </p>
                  </div>
                )}

                {error && (
                  <Alert
                    data-testid="error-alert"
                    className="border-red-500 bg-red-50"
                  >
                    <AlertCircle className="h-4 w-4 text-red-600" />
                    <AlertDescription className="text-red-800">
                      {error}
                    </AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert
                    data-testid="success-alert"
                    className="border-green-500 bg-green-50"
                  >
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-800">
                      {success}
                    </AlertDescription>
                  </Alert>
                )}

                <Button
                  data-testid="analyze-btn"
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="w-full bg-[#1A3C34] text-white hover:bg-[#1A3C34]/90 rounded-none border-b-2 border-[#D4FF32] px-8 py-6 uppercase tracking-wider transition-all active:translate-y-1 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Image"
                  )}
                </Button>

                {prediction && (
                  <Button
                    data-testid="new-scan-btn"
                    onClick={handleNewScan}
                    variant="outline"
                    className="w-full border-[#1A3C34] text-[#1A3C34] hover:bg-[#1A3C34] hover:text-white rounded-none px-8 py-6 uppercase tracking-wider"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    New Scan
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div>
            {prediction ? (
              <PredictionResult prediction={prediction} />
            ) : (
              <Card className="bg-white border border-[#1A3C34]/10 rounded-sm">
                <CardHeader>
                  <CardTitle
                    className="text-[#1A3C34]"
                    style={{ fontFamily: "'Playfair Display', serif" }}
                  >
                    Awaiting Analysis
                  </CardTitle>
                  <CardDescription className="text-[#5C6B66]">
                    Upload an image to see disease detection results
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-center py-20">
                  <div className="text-center text-[#8D9995]">
                    <Leaf className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <p>Results will appear here</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;

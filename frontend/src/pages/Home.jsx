import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Leaf, Scan, Activity, Microscope } from "lucide-react";
import { Button } from "@/components/ui/button";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F4F5F0]">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="relative min-h-screen bg-gradient-to-br from-[#1A3C34] to-[#0F231E] overflow-hidden"
      >
        {/* Background Image Overlay */}
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "url(https://images.pexels.com/photos/35494628/pexels-photo-35494628.jpeg)",
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        />

        <div className="relative z-10 container mx-auto px-4 py-20 flex flex-col items-center justify-center min-h-screen">
          <motion.div
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-center max-w-4xl"
          >
            <div className="flex items-center justify-center mb-6">
              <Leaf className="w-16 h-16 text-[#D4FF32]" />
            </div>
            <h1
              className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              AgriScan AI
            </h1>
            <p
              className="text-xl md:text-2xl text-white/90 mb-4"
              style={{ fontFamily: "'Manrope', sans-serif" }}
            >
              Advanced Plant Disease Detection
            </p>
            <p className="text-lg text-white/70 mb-12 max-w-2xl mx-auto leading-relaxed">
              Harness the power of machine learning to identify plant diseases
              instantly. Upload an image, get accurate diagnosis, and protect
              your crops.
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button
                data-testid="get-started-btn"
                onClick={() => navigate("/dashboard")}
                className="bg-[#1A3C34] text-white hover:bg-[#1A3C34]/90 rounded-none border-b-2 border-[#D4FF32] px-8 py-6 text-sm uppercase tracking-wider transition-all active:translate-y-1"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                <Scan className="mr-2 h-5 w-5" />
                Start Scanning
              </Button>
              <Button
                data-testid="view-history-btn"
                onClick={() => navigate("/history")}
                className="bg-transparent border-2 border-white text-white hover:bg-white hover:text-[#1A3C34] rounded-none px-8 py-6 text-sm uppercase tracking-wider transition-all"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                View History
              </Button>
            </div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 max-w-5xl w-full"
          >
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-sm p-6">
              <Microscope className="w-10 h-10 text-[#D4FF32] mb-4" />
              <h3
                className="text-xl font-semibold text-white mb-2"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                ML-Powered Analysis
              </h3>
              <p className="text-white/70 text-sm">
                Advanced machine learning models trained on thousands of plant
                disease images
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-sm p-6">
              <Activity className="w-10 h-10 text-[#D4FF32] mb-4" />
              <h3
                className="text-xl font-semibold text-white mb-2"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Real-Time Detection
              </h3>
              <p className="text-white/70 text-sm">
                Get instant disease identification with confidence scores in
                seconds
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-sm p-6">
              <Leaf className="w-10 h-10 text-[#D4FF32] mb-4" />
              <h3
                className="text-xl font-semibold text-white mb-2"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Multi-Crop Support
              </h3>
              <p className="text-white/70 text-sm">
                Supports tomato, potato, corn, pepper, grape, apple, and more
              </p>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* How It Works Section */}
      <div className="py-20 px-4 bg-[#F4F5F0]">
        <div className="container mx-auto max-w-6xl">
          <h2
            className="text-4xl md:text-5xl font-bold text-[#1A3C34] text-center mb-16"
            style={{ fontFamily: "'Playfair Display', serif" }}
          >
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="bg-white border border-[#1A3C34]/10 rounded-sm p-8 result-card"
            >
              <div
                className="text-6xl font-bold text-[#D4FF32] mb-4"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                01
              </div>
              <h3
                className="text-2xl font-semibold text-[#1A3C34] mb-3"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Upload Image
              </h3>
              <p className="text-[#5C6B66]">
                Take a photo of the affected plant or upload an existing image
                from your device
              </p>
            </motion.div>
            <motion.div
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="bg-white border border-[#1A3C34]/10 rounded-sm p-8 result-card"
            >
              <div
                className="text-6xl font-bold text-[#D4FF32] mb-4"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                02
              </div>
              <h3
                className="text-2xl font-semibold text-[#1A3C34] mb-3"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                AI Analysis
              </h3>
              <p className="text-[#5C6B66]">
                Our ML model analyzes the image and identifies potential
                diseases with confidence scores
              </p>
            </motion.div>
            <motion.div
              initial={{ y: 30, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-white border border-[#1A3C34]/10 rounded-sm p-8 result-card"
            >
              <div
                className="text-6xl font-bold text-[#D4FF32] mb-4"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                03
              </div>
              <h3
                className="text-2xl font-semibold text-[#1A3C34] mb-3"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Get Results
              </h3>
              <p className="text-[#5C6B66]">
                Receive detailed diagnosis with disease information and
                recommended actions
              </p>
            </motion.div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-20 px-4 bg-[#1A3C34]">
        <div className="container mx-auto max-w-4xl text-center">
          <h2
            className="text-4xl md:text-5xl font-bold text-white mb-6"
            style={{ fontFamily: "'Playfair Display', serif" }}
          >
            Ready to Protect Your Crops?
          </h2>
          <p className="text-xl text-white/80 mb-8">
            Start detecting plant diseases with AI-powered precision today
          </p>
          <Button
            data-testid="cta-start-btn"
            onClick={() => navigate("/dashboard")}
            className="bg-[#D4FF32] text-[#1A3C34] hover:bg-[#D4FF32]/90 rounded-none border-b-2 border-[#1A3C34] px-10 py-6 text-sm uppercase tracking-wider transition-all active:translate-y-1"
            style={{ fontFamily: "'JetBrains Mono', monospace" }}
          >
            Start Free Scan
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Home;

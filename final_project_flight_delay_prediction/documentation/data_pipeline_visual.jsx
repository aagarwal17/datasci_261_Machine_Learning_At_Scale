import React from 'react';
import { Database, Filter, Wrench, CheckCircle, TrendingUp, BarChart3, AlertTriangle } from 'lucide-react';

export default function DataPipelineVisual() {
  const stages = [
    {
      name: "Raw OTPW",
      icon: Database,
      rows: "5.8M",
      cols: 216,
      delay: "18.2%",
      color: "bg-slate-500",
      textColor: "text-slate-700",
      borderColor: "border-slate-300"
    },
    {
      name: "Joined Data",
      icon: Database,
      rows: "7.4M",
      cols: 75,
      delay: "18.4%",
      color: "bg-blue-500",
      textColor: "text-blue-700",
      borderColor: "border-blue-300"
    },
    {
      name: "Cleaned",
      icon: Filter,
      rows: "7.3M",
      cols: 75,
      delay: "18.6%",
      color: "bg-green-500",
      textColor: "text-green-700",
      borderColor: "border-green-300"
    },
    {
      name: "Engineered",
      icon: Wrench,
      rows: "7.3M",
      cols: 124,
      delay: "18.6%",
      color: "bg-purple-500",
      textColor: "text-purple-700",
      borderColor: "border-purple-300"
    },
    {
      name: "Final",
      icon: CheckCircle,
      rows: "7.3M",
      cols: 104,
      delay: "18.6%",
      color: "bg-emerald-500",
      textColor: "text-emerald-700",
      borderColor: "border-emerald-300"
    }
  ];

  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">
            Flight Delay Prediction Pipeline
          </h1>
          <p className="text-lg text-slate-600">
            Data Processing & Feature Engineering Journey
          </p>
        </div>

        {/* Pipeline Flow */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-slate-700 mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6" />
            Data Pipeline Progression
          </h2>
          
          <div className="flex items-center justify-between gap-2">
            {stages.map((stage, idx) => (
              <React.Fragment key={stage.name}>
                {/* Stage Card */}
                <div className={`flex-1 bg-white rounded-lg shadow-md border-2 ${stage.borderColor} p-4 hover:shadow-lg transition-shadow`}>
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`${stage.color} p-2 rounded-lg`}>
                      <stage.icon className="w-5 h-5 text-white" />
                    </div>
                    <h3 className="font-bold text-slate-800 text-sm">{stage.name}</h3>
                  </div>
                  
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Rows:</span>
                      <span className="font-semibold text-slate-800">{stage.rows}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Columns:</span>
                      <span className="font-semibold text-slate-800">{stage.cols}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Delay:</span>
                      <span className={`font-semibold ${stage.textColor}`}>{stage.delay}</span>
                    </div>
                  </div>
                </div>

                {/* Arrow */}
                {idx < stages.length - 1 && (
                  <div className="flex-shrink-0 text-slate-400">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          {/* Class Balance */}
          <div className="bg-white rounded-lg shadow-md p-6 border-2 border-amber-300">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-6 h-6 text-amber-600" />
              <h3 className="text-xl font-bold text-slate-800">Class Balance</h3>
            </div>
            
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600">On-Time Flights</span>
                  <span className="text-sm font-semibold text-slate-800">81.4%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-green-500 h-3 rounded-full" style={{width: '81.4%'}}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600">Delayed Flights</span>
                  <span className="text-sm font-semibold text-slate-800">18.6%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-3">
                  <div className="bg-red-500 h-3 rounded-full" style={{width: '18.6%'}}></div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Imbalance Ratio:</span>
                  <span className="text-lg font-bold text-amber-600">4.37:1</span>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Engineering */}
          <div className="bg-white rounded-lg shadow-md p-6 border-2 border-purple-300">
            <div className="flex items-center gap-2 mb-4">
              <Wrench className="w-6 h-6 text-purple-600" />
              <h3 className="text-xl font-bold text-slate-800">Feature Engineering</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-slate-200">
                <span className="text-sm text-slate-600">Original Features:</span>
                <span className="text-2xl font-bold text-slate-800">75</span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b border-slate-200">
                <span className="text-sm text-slate-600">Engineered Features:</span>
                <span className="text-2xl font-bold text-green-600">+29</span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b border-slate-200">
                <span className="text-sm text-slate-600">Features Removed:</span>
                <span className="text-2xl font-bold text-red-600">-20</span>
              </div>
              
              <div className="flex justify-between items-center pt-2">
                <span className="text-sm font-semibold text-slate-700">Final Count:</span>
                <span className="text-3xl font-bold text-purple-600">104</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500">
            <div className="text-sm text-slate-600 mb-1">Data Quality</div>
            <div className="text-2xl font-bold text-blue-600">High</div>
            <div className="text-xs text-slate-500 mt-1">163K rows cleaned (2.2%)</div>
            <div className="text-xs text-slate-500">0 nulls remaining</div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-emerald-500">
            <div className="text-sm text-slate-600 mb-1">Data Source</div>
            <div className="text-2xl font-bold text-emerald-600">2 Sources</div>
            <div className="text-xs text-slate-500 mt-1">BTS + NOAA Weather</div>
            <div className="text-xs text-slate-500">12 months (2015)</div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-4 border-l-4 border-amber-500">
            <div className="flex items-center gap-1 mb-1">
              <AlertTriangle className="w-4 h-4 text-amber-600" />
              <div className="text-sm text-slate-600">Requires</div>
            </div>
            <div className="text-lg font-bold text-amber-600">Imbalance Handling</div>
            <div className="text-xs text-slate-500 mt-1">SMOTE, class weights,</div>
            <div className="text-xs text-slate-500">or sampling strategies</div>
          </div>
        </div>
      </div>
    </div>
  );
}

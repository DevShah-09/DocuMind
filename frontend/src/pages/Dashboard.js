import { motion } from 'framer-motion';
import { FileText, Database, Activity, Clock, TrendingUp, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const stats = [
    { label: 'Total Documents', value: '1,247', icon: FileText, color: 'text-blue-600' },
    { label: 'Total Embeddings', value: '45.2K', icon: Database, color: 'text-cyan-600' },
    { label: 'Vector DB Status', value: 'Online', icon: Activity, color: 'text-green-600' },
    { label: 'Avg Response Time', value: '1.2s', icon: Clock, color: 'text-orange-600' },
    { label: 'Queries Today', value: '89', icon: TrendingUp, color: 'text-purple-600' },
  ];

  const queryData = [
    { date: 'Mon', queries: 65 },
    { date: 'Tue', queries: 78 },
    { date: 'Wed', queries: 90 },
    { date: 'Thu', queries: 81 },
    { date: 'Fri', queries: 95 },
    { date: 'Sat', queries: 72 },
    { date: 'Sun', queries: 89 },
  ];

  const collectionData = [
    { name: 'Research Papers', queries: 145 },
    { name: 'Documentation', queries: 98 },
    { name: 'Legal Documents', queries: 76 },
    { name: 'Technical Specs', queries: 54 },
  ];

  return (
    <div className="px-8 py-12" data-testid="dashboard">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
            Dashboard
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Overview of your document intelligence system
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-12">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="hover:shadow-lg transition-all duration-200 border-slate-200 dark:border-slate-800" data-testid={`stat-card-${index}`}>
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">{stat.label}</p>
                        <p className="text-3xl font-bold text-slate-900 dark:text-white">{stat.value}</p>
                      </div>
                      <div className={`p-3 rounded-lg bg-slate-50 dark:bg-slate-800 ${stat.color}`}>
                        <Icon className="w-6 h-6" strokeWidth={1.5} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Line Chart */}
          <Card className="border-slate-200 dark:border-slate-800" data-testid="query-activity-chart">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-500" strokeWidth={2} />
                Query Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={queryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip />
                  <Line type="monotone" dataKey="queries" stroke="#06b6d4" strokeWidth={3} dot={{ fill: '#06b6d4', r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Bar Chart */}
          <Card className="border-slate-200 dark:border-slate-800" data-testid="top-collections-chart">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-slate-700" strokeWidth={2} />
                Top Collections
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={collectionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip />
                  <Bar dataKey="queries" fill="#0f172a" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
//import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.Remove;


public class Classification {
	final static String gesture = "Brushing Teeth";
	final static Double time_interval = 1.0;

	/*final static String[] header = (//"Avg Jerk X,Avg Jerk Y,Avg Jerk Z,"
			//+ "Avg Height X,Avg Height Y,Avg Height Z,"
			//+ "Stdev Height X,Stdev Height Y,Stdev Height Z,"
			//+ "Avg Dist to Mean X,Avg Dist to Mean Y,Avg Dist to Mean Z,"
			//+ "Stdev to Mean X,Stdev to Mean Y,Stdev to Mean Z,"
			 "Energy X,Energy Y,Energy Z,"
			//+  "Entropy Z,"
			+ "Average X,Average Y,Average Z,"
			//+ "Average XZ,Average YZ,"
			+ "Standard Deviation X,Standard Deviation Y,Standard Deviation Z,"
			+ "Correlation XY,Correlation XZ,Correlation YZ,"
			//+ "RMS X,RMS Y,RMS Z,"
			//+ "Axis Order YZ,"
			//+ "Num Peaks Y,"
			//+ "Average Peaks X,Average Peaks Z,"
			//+ "Standard Deviation Peaks X,Standard Deviation Peaks Y,Standard Deviation Peaks Z,"
			//+ "Num Valleys X,Num Valleys Y,"
			//+ "Average Valleys X,Average Valleys Y,Average Valleys Z,"
			//+ "Standard Deviation Valleys X,Standard Deviation Valleys Y,Standard Deviation Valleys Z,"
			+ "Gesture,").split(",");
	//+ "Start Time,End Time").split(",");*/

	final static String[] header = (  "Avg Jerk X,Avg Jerk Y,Avg Jerk Z,"
			+ "Avg Height X,Avg Height Y,Avg Height Z,"
			+ "Stdev Height X,Stdev Height Y,Stdev Height Z,"
			+ "Avg Dist to Mean X,Avg Dist to Mean Y,Avg Dist to Mean Z,"
			+ "Stdev to Mean X,Stdev to Mean Y,Stdev to Mean Z,"
			+ "Energy X,Energy Y,Energy Z,"
			+ "Entropy X,Entropy Y,Entropy Z,"
			+ "Average X,Average Y,Average Z,"
			+ "Average XY,Average XZ,Average YZ,"
			+ "Standard Deviation X,Standard Deviation Y,Standard Deviation Z,"
			+ "Correlation XY,Correlation XZ,Correlation YZ,"
			+ "RMS X,RMS Y,RMS Z,"
			+ "Axis Order XY,Axis Order XZ,Axis Order YZ,"
			+ "Num Peaks X,Num Peaks Y,Num Peaks Z,"
			+ "Average Peaks X,Average Peaks Y,Average Peaks Z,"
			+ "Standard Deviation Peaks X,Standard Deviation Peaks Y,Standard Deviation Peaks Z,"
			+ "Num Valleys X,Num Valleys Y,Num Valleys Z,"
			+ "Average Valleys X,Average Valleys Y,Average Valleys Z,"
			+ "Standard Deviation Valleys X,Standard Deviation Valleys Y,Standard Deviation Valleys Z,"
			+ "Gesture,").split(",");
	//+ "Start Time,End Time").split(",");

	public static void weka_classify (String testData, String trainData, String output) {
		DataSource source;
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		AttributeSelection filter = new AttributeSelection();
		Remove remove = new Remove();

		try {
			source = new DataSource(trainData);
			Instances trainDataset = source.getDataSet();

			ArrayList<Integer> index_remove = new ArrayList<Integer>();
			for (int i = 0; i <= trainDataset.numAttributes()-1; i++) {
				if (!Arrays.asList(header).contains(trainDataset.attribute(i).name())) {
					index_remove.add(i);
				}
			}
			/*int numClasses = trainDataset.numClasses();
			for (int i = 0; i < numClasses; i++) {
				String classValue = trainDataset.classAttribute().value(i);
				System.out.println("Class Value " + i + " is " + classValue);
			}*/

			//create array of attributes to keep

			remove.setAttributeIndicesArray(index_remove.stream().mapToInt(i -> i).toArray());
			remove.setInvertSelection(false);
			remove.setInputFormat(trainDataset);
			trainDataset = Filter.useFilter(trainDataset, remove);
			trainDataset.setClassIndex(trainDataset.numAttributes()-1);


			/*SpreadSubsample ff = new SpreadSubsample();
			String opt = "-M 1";
			String[] optArray = weka.core.Utils.splitOptions(opt);
			ff.setOptions(optArray);
			ff.setInputFormat(trainDataset);
			Instances filteredInstances = Filter.useFilter(trainDataset, ff);*/

			eval.setMissingSeparate(true);
			filter.setEvaluator(eval);
			filter.setSearch(search);
			// Apply filter
			filter.setInputFormat(trainDataset);

			trainDataset = Filter.useFilter(trainDataset, filter);
			String modelfile = "data/thesis/models/mp_new.model";
			File file = new File(modelfile);
			//String[] options = weka.core.Utils.splitOptions("-K 3");
			Classifier cls = new MultilayerPerceptron();
			//cls.setOptions(options);
			if (!file.exists()) {


				System.out.println("\t" + "Building Model....");

				cls.buildClassifier(trainDataset);
				for (int i = 0; i < trainDataset.numAttributes(); i++) {
					System.out.println(trainDataset.attribute(i));
				}
				Debug.saveToFile(modelfile, cls);
			} else {
				cls = (Classifier) SerializationHelper.read(modelfile);
			}
			//System.out.println(cls.toString());


			System.out.println("\t" + "Done");
			System.out.println("\t" + "Testing...");
			DataSource source1 = new DataSource(testData);
			Instances testDataset = source1.getDataSet();

			ArrayList<String> feature_subset = new ArrayList<>();
			for (int i = 0; i < trainDataset.numAttributes(); i++) {
				feature_subset.add(trainDataset.attribute(i).name());
				//System.out.println(trainDataset.attribute(i));
			}

			index_remove.clear();
			for (int a = 0; a <= testDataset.numAttributes()-1; a++) {
				if (!feature_subset.contains(testDataset.attribute(a).name())) {
					index_remove.add(a);
				} else {
					//System.out.println(testDataset.attribute(a).name());
				}
			}

			remove.setAttributeIndicesArray(index_remove.stream().mapToInt(i -> i).toArray());
			remove.setInvertSelection(false);
			remove.setInputFormat(testDataset);
			testDataset = Filter.useFilter(testDataset, remove);
			testDataset.setClassIndex(testDataset.numAttributes()-1);
			PrintWriter out = new PrintWriter(output);
			System.out.println("\t" + "Done");
			out.println("============,==============");
			out.println("Actual Class, RF Predicted");
			for (int i = 0; i < testDataset.numInstances(); i++) {
				//get class double value for current instance
				double actualClass = testDataset.instance(i).classValue();
				//get class string value using the class index using the class's int value
				String actual = testDataset.classAttribute().value((int) actualClass);
				//get Instance object of current instance
				Instance newInst = testDataset.instance(i);
				//System.out.println (newInst);
				//call classifyInstance, which returns a double value for the class
				try {
					double predRF = cls.classifyInstance(newInst);
					//use this value to get string value of the predicted class
					String predString = testDataset.classAttribute().value((int) predRF);
					//System.out.println(predString);
					out.println(actual + ", " + predString);
				} catch (ArrayIndexOutOfBoundsException e) {
					e.printStackTrace();
				}

			}
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void algorithm_test(String testData, String output, String falsepos_path, String gesture_name) {
		CSVReader reader = null;
		String[] line = null;
		String[] header = null;
		int activity =  52;
		PrintWriter out = null;
		PrintWriter fp = null;
		ArrayList<String> path = null;
		Iterator<String> it = null;
		int line_num = 1;
		String path_string = "";
		System.out.println(gesture_name);
		if (testData.contains(gesture_name)) {
			try {
				reader = new CSVReader(new FileReader(testData));
				header = reader.readNext();
				System.out.println("\t" + "Testing...");
				for (int key = 0; key < header.length; key++) {
					if (header[key].matches("Gesture")) {
						activity = key;
					}
				}

				out = new PrintWriter(output);
				fp = new PrintWriter(falsepos_path);
				out.println("============,==============");
				out.println("Actual Class, Predicted");
				fp.println("Line Number, Path");
				//out.println(reader.readNext()[activity] + "," + j48test(line));
				while ((line = reader.readNext()) != null) {
					line_num++;
					path = j48_washhands(line,testData);
					String actual = line[activity];
					String predicted = path.get(path.size()-1);
					out.println(actual + "," + predicted);
					if (!predicted.matches(actual) || actual.contains("Washing Hands")) {
						it = path.iterator();
						path_string = "";
						while(it.hasNext()) {
							path_string = path_string + "," + it.next();
						}
						//path_string = path_string.substring(path_string.length()-2);
						fp.println(line_num + path_string);
					}
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				try {
					reader.close();
					out.close();
					fp.close();
					System.out.println("\t" + "Done");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}

	public static void classify_gesture(String output) {
		BufferedReader br = null;
		String line = "";
		Double gesture_time = 0.0;
		Double inactive_time = 0.0;
		int line_num = 1;
		int gesturegroup_cnt = 0;
		int max_cnt = 0;

		try {
			br = new BufferedReader(new FileReader(output));
			br.readLine();
			while ((line = br.readLine()) != null) {
				line_num++;
				/* result[0] - actual
				   result[1] - predicted */
				String[] result = line.split(",");
				String predicted = result[1].trim();

				if (predicted.equalsIgnoreCase(gesture)) {

					gesturegroup_cnt++;
					if (gesture_time == 0) {
						gesture_time += time_interval;
					} else {
						gesture_time += time_interval + inactive_time;
					}
					inactive_time = 0.0;
					if(gesture_time >= 75) {
						if (max_cnt >= 25) {
							System.out.println(line_num);
							System.out.println("Brushed Teeth!");
						}
					}
				} else {
					inactive_time += time_interval;
					max_cnt = Math.max(max_cnt, gesturegroup_cnt);
					gesturegroup_cnt = 0;
					if (inactive_time >= 15) {
						gesture_time = 0.0;
						max_cnt = 0;
					}
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br !=null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static ArrayList<Double> accuracy(String output, String gesture) {
		BufferedReader br = null;
		String line = "";
		double num_lines = 0;
		double total_lines = 0;
		double num_correct = 0;
		double num_gest_correct = 0;
		ArrayList<Double> falseneg = new ArrayList<Double>();
		ArrayList<String> lines = new ArrayList<String>();
		try {
			br = new BufferedReader(new FileReader(output));
			line = br.readLine();
			line = br.readLine();
			lines.add(line);
			while ((line = br.readLine()) != null) {
				lines.add(line);
				total_lines++;
				String[] act_pred = line.split("\\s*,\\s*");
				if (act_pred[0].contains(gesture)) {
					num_lines++;
					if (act_pred[0].matches(act_pred[1])) {
						num_correct++;
						num_gest_correct++;
					} else {
						//System.out.println("Incorrect line: " + (total_lines-1));
						falseneg.add(total_lines-1);
					}
				} else {
					if (!act_pred[0].matches(act_pred[1])) {
						falseneg.add(total_lines-1);
					} else {
						num_correct++;
					}
				}
			}
			double accuracy = num_correct/total_lines * 100;
			double gesture_accuracy = num_gest_correct/num_lines * 100;
			if (num_lines > 60) {
				System.out.println("\t" + "Total Accuracy: " + accuracy  + " (" + num_correct + " out of " + total_lines + ")");
				System.out.println("\t" + "Gesture Accuracy: " + gesture_accuracy  + " (" + num_gest_correct + " out of " + num_lines + ")");
			}
			lines.add(0, "Accuracy:," + accuracy + " (" + num_correct + " out of " + total_lines + "),Gesture Accuracy:," + gesture_accuracy + "(" + num_gest_correct + " out of " + num_lines + ")");

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
					BufferedWriter out = new BufferedWriter(new FileWriter(output, false));
					Iterator<String> it = lines.iterator();
					while(it.hasNext()) {
						out.write(it.next()+"\n");
					}
					out.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		return falseneg;
	}

	public static ArrayList<Integer> cross_validation(String trainData) {

		DataSource source;
		ArrayList<Integer> incorrect = new ArrayList<Integer>();

		try {
			System.out.println("Building Model....");
			source = new DataSource(trainData);
			Instances trainDataset = source.getDataSet();
			trainDataset.setClassIndex(trainDataset.numAttributes()-1);

			int runs = 1;
			int folds = 10;

			RandomForest rf = new RandomForest();
			Classifier cls = (Classifier) rf;
			for (int i = 0; i < runs; i++) {
				// randomize data
				int seed = i + 1;
				Random rand = new Random(seed);
				Instances randData = new Instances(trainDataset);

				randData.randomize(rand);
				if (randData.classAttribute().isNominal())
					randData.stratify(folds);
				Instances train = null;
				Evaluation eval = new Evaluation(randData);
				ArrayList<Prediction> predictions = null;
				for (int n = 0; n < folds; n++) {
					System.out.println("Evaluating fold " + n);
					train = randData.trainCV(folds, n);
					Instances test = randData.testCV(folds, n);
					// the above code is used by the StratifiedRemoveFolds filter, the
					// code below by the Explorer/Experimenter:
					// Instances train = randData.trainCV(folds, n, rand);

					// build and evaluate classifier
					Classifier clsCopy = RandomForest.makeCopy(cls);
					clsCopy.buildClassifier(train);
					eval.evaluateModel(clsCopy, test);
				}
				System.out.println(train.size());
				for (int a = 0; a < train.size(); a++) {
					//Instance instance = train.get(a);
					predictions = eval.predictions();
					Prediction prediction = predictions.get(a);
					//System.out.println("Actual: " + prediction.actual());
					//System.out.println("Predicted: " + prediction.predicted());
					if (prediction.actual() != prediction.predicted()) {
						incorrect.add(a+1);
						//System.out.println(instance);
					}
				}

				// output evaluation
				System.out.println();
				System.out.println("=== Setup run " + (i+1) + " ===");
				System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(((RandomForest) cls).getOptions()));
				System.out.println("Dataset: " + trainDataset.relationName());
				System.out.println("Folds: " + folds);
				System.out.println("Seed: " + seed);
				System.out.println();
				System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return incorrect;
	}

	/*
	 * For each found False Positive and True Positive (predicted not actual)
	 * make list of paths, how often they occur, and their percentage of all the paths found in that FP or TP
	 * Probably use HashMap for this
	 * do the above first, see if there is just one or multiple common paths for TP but not FP
	 * if paths have a majority (greater than some percentage) that is 
	 * a common path for TPs but not FPs its a TP, 
	 * if a different majority a FP, 
	 * if not majority then remain what it is....?
	 */
	public static void tree_analysis(String folder, String stats_folder) {
		CSVReader reader = null;
		BufferedReader fp_reader = null;
		BufferedWriter fp_analysis_bw = null;
		BufferedWriter fp_total_analysis_bw = null;
		BufferedWriter TP_FP_bw = null;
		String[] line = null;
		String fp_line = null;
		String start_time;
		String end_time;
		int total_time;
		String type;
		String false_pos_file;
		Boolean record_path = false;
		String[] path;
		HashMap<String, Integer> path_count = new HashMap<String, Integer>();
		HashMap<String, Integer> TP_breakdown = new HashMap<String, Integer>();
		HashMap<String, Integer> FP_breakdown = new HashMap<String, Integer>();
		HashMap<String, Integer[]> TP_FP_breakdown = new HashMap<String, Integer[]>();


		int num_FP = 0;
		int num_TP = 0;
		//key, value

		try {
			fp_analysis_bw = new BufferedWriter(new FileWriter(folder + "/FP_analysis.csv"));
			fp_analysis_bw.write("File Name,TP/FP,Start Time,End Time,Count,Percentage,Path");
			fp_analysis_bw.write("\n");

			fp_total_analysis_bw = new BufferedWriter(new FileWriter(folder + "/FP_total_analysis.csv"));
			fp_total_analysis_bw.write("TP/FP,Count,Percentage,Path");
			fp_total_analysis_bw.newLine();

			TP_FP_bw = new BufferedWriter(new FileWriter(folder + "/TP_FP.csv"));
			TP_FP_bw.write("TP Count,FP Count,TP (%),FP (%),Path");
			TP_FP_bw.newLine();

			for (File stats_file : new File(folder + "/" + stats_folder).listFiles()) {
				try {
					reader = new CSVReader(new FileReader(stats_file));
					System.out.println(stats_file);
					while ((line = reader.readNext()) != null) {
						path_count.clear();

						if (line[0].contains("Gesture Start Time") || line[0].contains("False Positive Start Time")) {

							if (line[0].contains("Gesture")) {
								type = "TP";
							} else {
								type = "FP";
							}

							start_time = line[1];
							line = reader.readNext();
							end_time = line[1];
							total_time = new Double(Double.parseDouble(end_time) - Double.parseDouble(start_time) + 1).intValue();
							false_pos_file = stats_file.getName().substring(0, stats_file.getName().length()-4) + "_FP.csv";
							fp_reader = new BufferedReader(new FileReader(new File(folder + "/False Positive Paths/" + false_pos_file)));
							while ((fp_line = fp_reader.readLine()) != null) {
								path = fp_line.split(",",2);
								if (path[0].matches(start_time)) {

									fp_analysis_bw.newLine();
									record_path = true;
								}

								if (record_path) {
									if (path_count.containsKey(path[1])) {
										path_count.put(path[1],path_count.get(path[1]) + 1);
									} else {
										path_count.put(path[1], 1);									
									}

									if (type.matches("TP")) {
										num_TP++;

										if (TP_breakdown.containsKey(path[1])) {
											TP_breakdown.put(path[1],TP_breakdown.get(path[1]) + 1);
										} else {
											TP_breakdown.put(path[1], 1);									
										}

										if (TP_FP_breakdown.containsKey(path[1])) {
											TP_FP_breakdown.put(path[1], new Integer[] {TP_FP_breakdown.get(path[1])[0]+1, TP_FP_breakdown.get(path[1])[1]});
										} else {
											TP_FP_breakdown.put(path[1], new Integer[] {1, 0});
										}
									} else {
										num_FP++;

										if (FP_breakdown.containsKey(path[1])) {
											FP_breakdown.put(path[1],FP_breakdown.get(path[1]) + 1);
										} else {
											FP_breakdown.put(path[1], 1);									
										}

										if (TP_FP_breakdown.containsKey(path[1])) {
											TP_FP_breakdown.put(path[1], new Integer[] {TP_FP_breakdown.get(path[1])[0], TP_FP_breakdown.get(path[1])[1]+1});
										} else {
											TP_FP_breakdown.put(path[1], new Integer[] {0, 1});
										}
									}
								}

								if (path[0].matches(end_time)) {
									Iterator<Entry<String, Integer>> path_it = path_count.entrySet().iterator();
									while (path_it.hasNext()) {
										Map.Entry<String, Integer> path_inst = (Map.Entry<String, Integer>) path_it.next();
										Double percentage = (new Double(path_inst.getValue())/new Double(total_time))*100;
										fp_analysis_bw.write(stats_file.getName() + "," + type + "," + start_time + "," + end_time + "," + path_inst.getValue() + "," + percentage + "," + path_inst.getKey());
										fp_analysis_bw.write("\n");

									}
									record_path = false;
								}
							}	
						}
					}

				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					if (reader != null) {
						try {
							reader.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}

			fp_analysis_bw.close();

			//fp_total_analysis
			Iterator<Entry<String, Integer>> FP_it = FP_breakdown.entrySet().iterator();			
			while (FP_it.hasNext()) {
				Map.Entry<String, Integer> path_inst = (Map.Entry<String, Integer>) FP_it.next();
				Double percentage = (new Double(path_inst.getValue())/new Double(num_FP))*100;
				fp_total_analysis_bw.write("FP," + path_inst.getValue() + "," + percentage + "," + path_inst.getKey());
				fp_total_analysis_bw.write("\n");
			}

			fp_total_analysis_bw.write("\n");

			Iterator<Entry<String, Integer>> TP_it = TP_breakdown.entrySet().iterator();			
			while (TP_it.hasNext()) {
				Map.Entry<String, Integer> path_inst = (Map.Entry<String, Integer>) TP_it.next();
				Double percentage = (new Double(path_inst.getValue())/new Double(num_TP))*100;
				fp_total_analysis_bw.write("TP," + path_inst.getValue() + "," + percentage + "," + path_inst.getKey());
				fp_total_analysis_bw.write("\n");
			}

			fp_total_analysis_bw.close();

			Iterator<Entry<String, Integer[]>> TP_FP_it = TP_FP_breakdown.entrySet().iterator();
			while (TP_FP_it.hasNext()) {
				Map.Entry<String, Integer[]> path_inst = (Map.Entry<String, Integer[]>) TP_FP_it.next();
				Double total = new Double(path_inst.getValue()[0]+path_inst.getValue()[1]);

				Double TP_percentage = (new Double(path_inst.getValue()[0])/total)*100;
				Double FP_percentage = (new Double(path_inst.getValue()[1])/total)*100;

				TP_FP_bw.write(path_inst.getValue()[0] + "," + path_inst.getValue()[1] + "," + TP_percentage + "," + FP_percentage + "," + path_inst.getKey());
				TP_FP_bw.write("\n");
			}

			TP_FP_bw.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static ArrayList<String> j48_brushteeth (String[] line, String filename) {
		Double avg_jerkX = Double.parseDouble(line[0]);
		Double avg_jerkY = Double.parseDouble(line[1]);
		Double avg_jerkZ = Double.parseDouble(line[2]);
		Double avg_heightX = Double.parseDouble(line[3]);
		Double avg_heightY = Double.parseDouble(line[4]);
		Double avg_heightZ = Double.parseDouble(line[5]);
		Double stdev_heightX = Double.parseDouble(line[6]);
		Double stdev_heightY = Double.parseDouble(line[7]);
		Double stdev_heightZ = Double.parseDouble(line[8]);
		Double avg_dist_meanX = Double.parseDouble(line[10]);
		Double avg_dist_meanY = Double.parseDouble(line[11]);
		Double avg_dist_meanZ = Double.parseDouble(line[12]);
		Double stdev_meanX = Double.parseDouble(line[13]);
		Double stdev_meanY = Double.parseDouble(line[14]);
		Double stdev_meanZ = Double.parseDouble(line[15]);
		Double energyX = Double.parseDouble(line[9]);
		Double energyY = Double.parseDouble(line[10]);
		Double energyZ = Double.parseDouble(line[11]);
		Double entropyX = Double.parseDouble(line[12]);
		Double entropyY = Double.parseDouble(line[13]);
		Double entropyZ = Double.parseDouble(line[14]);
		Double avgX = Double.parseDouble(line[15]);
		Double avgY = Double.parseDouble(line[16]);
		Double avgZ = Double.parseDouble(line[17]);
		Double avgXY = Double.parseDouble(line[18]);
		Double avgXZ = Double.parseDouble(line[19]);
		Double avgYZ = Double.parseDouble(line[20]);
		Double stdevX = Double.parseDouble(line[21]);
		Double stdevY = Double.parseDouble(line[22]);
		Double stdevZ = Double.parseDouble(line[23]);
		Double corrXY = Double.parseDouble(line[24]);
		Double corrXZ = Double.parseDouble(line[25]);
		Double corrYZ = Double.parseDouble(line[26]);
		Double rmsX = Double.parseDouble(line[27]);
		Double rmsY = Double.parseDouble(line[28]);
		Double rmsZ = Double.parseDouble(line[29]);
		Double axis_orderXY = Double.parseDouble(line[30]);
		Double axis_orderXZ = Double.parseDouble(line[31]);
		Double axis_orderYZ = Double.parseDouble(line[32]);
		Double num_peaksX = Double.parseDouble(line[33]);
		Double num_peaksY = Double.parseDouble(line[34]);
		Double num_peaksZ = Double.parseDouble(line[35]);
		Double avg_peaksX = Double.parseDouble(line[36]);
		Double avg_peaksY = Double.parseDouble(line[37]);
		Double avg_peaksZ = Double.parseDouble(line[38]);
		Double stdev_peaksX = Double.parseDouble(line[39]);
		Double stdev_peaksY = Double.parseDouble(line[40]);
		Double stdev_peaksZ = Double.parseDouble(line[41]);
		Double num_valleysX = Double.parseDouble(line[42]);
		Double num_valleysY = Double.parseDouble(line[43]);
		Double num_valleysZ = Double.parseDouble(line[44]);
		Double avg_valleysX = Double.parseDouble(line[45]);
		Double avg_valleysY = Double.parseDouble(line[46]);
		Double avg_valleysZ = Double.parseDouble(line[47]);
		Double stdev_valleysX = Double.parseDouble(line[48]);
		Double stdev_valleysY = Double.parseDouble(line[49]);
		Double stdev_valleysZ = Double.parseDouble(line[50]);
		//Double zero_crossingsX = Double.parseDouble(line[51]);
		//Double zero_crossingsY = Double.parseDouble(line[52]);
		//Double zero_crossingsZ = Double.parseDouble(line[53]);
		//Double num_points = Double.parseDouble(line[54]);
		//String gesture = line[51];
		//Double start_time = Double.parseDouble(line[52]);
		//Double end_time = Double.parseDouble(line[53]);
		ArrayList<String> path = new ArrayList<String>();
		if (avg_peaksZ <= 669.333333) {
			path.add("Average Peaks Z <= 669.333333");
			if (avgYZ <= 5028) {
				path.add("Average YZ <= 5028");
				if (avg_valleysX <= -304) {
					path.add("Average Valleys X <= -304");
					if (stdevZ <= 379.821559) {
						path.add("Standard Deviation Z <= 379.821559");
						if (energyZ <= 361216) {
							path.add("Energy Z <= 361216");
							if (avgYZ <= 347.404255) {
								path.add("Average YZ <= 347.404255");
								path.add("Inactive");
								return path;
							} else {
								path.add("Average YZ > 347.404255");
								if (avgX <= -967.652174) {
									path.add("Average X <= -967.652174");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Average X > -967.652174");
									if (corrXY <= -391729.6327) {
										path.add("Correlation XY <= -391729.6327");
										if (energyY <= 10000000) {
											path.add("Energy Y <= 10000000");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Energy Y > 10000000");
											if (avg_peaksX <= -444) {
												path.add("Average Peaks X <= -444");
												if (avg_peaksY <= 949.333333) {
													path.add("Average Peaks Y <= 949.333333");
													path.add("Inactive");
													return path;
												} else {
													path.add("Average Peaks Y > 949.333333");
													if (avg_peaksX <= -460) {
														path.add("Average Peaks X <= -460");
														path.add("Washing Hands");
														return path;
													} else {
														path.add("Average Peaks X > -460");
														path.add("Inactive");
														return path;
													}
												}
											} else {
												path.add("Average Peaks X > -444");
												if (avg_jerkZ <= -32.48198) {
													path.add("Avg Jerk Z <= -32.48198");
													path.add("Inactive");
													return path;
												} else {
													path.add("Avg Jerk Z > -32.48198");
													path.add("Washing Hands");
													return path;
												}
											}
										}
									} else {
										path.add("Correlation XY > -391729.6327");
										path.add("Inactive");
										return path;
									}
								}
							}
						} else {
							path.add("Energy Z > 361216");
							if (corrXY <= -419083.6364) {
								path.add("Correlation XY <= -419083.6364");
								if (avg_peaksZ <= 468) {
									path.add("Average Peaks Z <= 468");
									if (avg_heightX <= 496) {
										path.add("Avg Height X <= 496");
										if (energyY <= 35300000) {
											path.add("Energy Y <= 35300000");
											if (stdev_valleysX <= 3.771236) {
												path.add("Standard Deviation Valleys X <= 3.771236");
												path.add("Inactive");
												return path;
											} else {
												path.add("Standard Deviation Valleys X > 3.771236");
												if (num_valleysX <= 2) {
													path.add("Num Valleys X <= 2");
													if (avg_jerkZ <= 6.73653) {
														path.add("Avg Jerk Z <= 6.73653");
														if (stdev_peaksZ <= 7.228063) {
															path.add("Standard Deviation Peaks Z <= 7.228063");
															path.add("Inactive");
															return path;
														} else {
															path.add("Standard Deviation Peaks Z > 7.228063");
															if (avg_heightY <= 210) {
																path.add("Avg Height Y <= 210");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Avg Height Y > 210");
																path.add("Inactive");
																return path;
															}
														}
													} else {
														path.add("Avg Jerk Z > 6.73653");
														if (avg_valleysY <= 452) {
															path.add("Average Valleys Y <= 452");
															path.add("Inactive");
															return path;
														} else {
															path.add("Average Valleys Y > 452");
															path.add("Washing Hands");
															return path;
														}
													}
												} else {
													path.add("Num Valleys X > 2");
													path.add("Inactive");
													return path;
												}
											}
										} else {
											path.add("Energy Y > 35300000");
											path.add("Inactive");
											return path;
										}
									} else {
										path.add("Avg Height X > 496");
										if (energyX <= 23200000) {
											path.add("Energy X <= 23200000");
											if (avg_valleysX <= -824) {
												path.add("Average Valleys X <= -824");
												path.add("Washing Hands");
												return path;
											} else {
												path.add("Average Valleys X > -824");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("Energy X > 23200000");
											if (axis_orderYZ <= 1) {
												path.add("Axis Order YZ <= 1");
												path.add("Inactive");
												return path;
											} else {
												path.add("Axis Order YZ > 1");
												path.add("Washing Hands");
												return path;
											}
										}
									}
								} else {
									path.add("Average Peaks Z > 468");
									if (rmsX <= 719.188915) {
										path.add("RMS X <= 719.188915");
										if (avg_valleysX <= -572) {
											path.add("Average Valleys X <= -572");
											if (num_peaksX <= 0) {
												path.add("Num Peaks X <= 0");
												if (avg_heightZ <= 412) {
													path.add("Avg Height Z <= 412");
													path.add("Inactive");
													return path;
												} else {
													path.add("Avg Height Z > 412");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Num Peaks X > 0");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Average Valleys X > -572");
											path.add("Inactive");
											return path;
										}
									} else {
										path.add("RMS X > 719.188915");
										path.add("Inactive");
										return path;
									}
								}
							} else {
								path.add("Correlation XY > -419083.6364");
								if (stdevY <= 77.195663) {
									path.add("Standard Deviation Y <= 77.195663");
									if (avgZ <= -893.061224) {
										path.add("Average Z <= -893.061224");
										if (avg_peaksX <= -304) {
											path.add("Average Peaks X <= -304");
											if (energyY <= 260160) {
												path.add("Energy Y <= 260160");
												path.add("Washing Hands");
												return path;
											} else {
												path.add("Energy Y > 260160");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("Average Peaks X > -304");
											path.add("Inactive");
											return path;
										}
									} else {
										path.add("Average Z > -893.061224");
										if (avg_heightX <= 334.666667) {
											path.add("Avg Height X <= 334.666667");
											path.add("Inactive");
											return path;
										} else {
											path.add("Avg Height X > 334.666667");
											if (num_valleysZ <= 0) {
												path.add("Num Valleys Z <= 0");
												path.add("Washing Hands");
												return path;
											} else {
												path.add("Num Valleys Z > 0");
												path.add("Inactive");
												return path;
											}
										}
									}
								} else {
									path.add("Standard Deviation Y > 77.195663");
									if (avgY <= 955.333333) {
										path.add("Average Y <= 955.333333");
										if (corrXZ <= 222585.0435) {
											path.add("Correlation XZ <= 222585.0435");
											if (avgXZ <= -880.666667) {
												path.add("Average XZ <= -880.666667");
												if (stdev_valleysY <= 33.941125) {
													path.add("Standard Deviation Valleys Y <= 33.941125");
													if (num_peaksZ <= 0) {
														path.add("Num Peaks Z <= 0");
														if (avg_peaksX <= -952) {
															path.add("Average Peaks X <= -952");
															path.add("Washing Hands");
															return path;
														} else {
															path.add("Average Peaks X > -952");
															if (entropyX <= 2.68566) {
																path.add("Entropy X <= 2.68566");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Entropy X > 2.68566");
																path.add("Inactive");
																return path;
															}
														}
													} else {
														path.add("Num Peaks Z > 0");
														path.add("Inactive");
														return path;
													}
												} else {
													path.add("Standard Deviation Valleys Y > 33.941125");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Average XZ > -880.666667");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("Correlation XZ > 222585.0435");
											if (avgYZ <= 759) {
												path.add("Average YZ <= 759");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average YZ > 759");
												if (stdev_valleysZ <= 25.095019) {
													path.add("Standard Deviation Valleys Z <= 25.095019");
													if (entropyZ <= 3.894698) {
														path.add("Entropy Z <= 3.894698");
														path.add("Inactive");
														return path;
													} else {
														path.add("Entropy Z > 3.894698");
														if (axis_orderYZ <= 0) {
															path.add("Axis Order YZ <= 0");
															if (num_valleysZ <= 0) {
																path.add("Num Valleys Z <= 0");
																if (avg_heightZ <= 28) {
																	path.add("Avg Height Z <= 28");
																	path.add("Inactive");
																	return path;
																} else {
																	path.add("Avg Height Z > 28");
																	if (avg_jerkX <= -90.00246) {
																		path.add("Avg Jerk X <= -90.00246");
																		path.add("Washing Hands");
																		return path;
																	} else {
																		path.add("Avg Jerk X > -90.00246");
																		path.add("Inactive");
																		return path;
																	}
																}
															} else {
																path.add("Num Valleys Z > 0");
																if (num_peaksY <= 0) {
																	path.add("Num Peaks Y <= 0");
																	path.add("Inactive");
																	return path;
																} else {
																	path.add("Num Peaks Y > 0");
																	if (stdevX <= 104.648171) {
																		path.add("Standard Deviation X <= 104.648171");
																		if (stdev_heightZ <= 4) {
																			path.add("Stdev Height Z <= 4");
																			path.add("Washing Hands");
																			return path;
																		} else {
																			path.add("Stdev Height Z > 4");
																			path.add("Inactive");
																			return path;
																		}
																	} else {
																		path.add("Standard Deviation X > 104.648171");
																		if (entropyY <= 8.748202) {
																			path.add("Entropy Y <= 8.748202");
																			if (stdevZ <= 94.08729) {
																				path.add("Standard Deviation Z <= 94.08729");
																				if (energyZ <= 13100000) {
																					path.add("Energy Z <= 13100000");
																					path.add("Washing Hands");
																					return path;
																				} else {
																					path.add("Energy Z > 13100000");
																					path.add("Inactive");
																					return path;
																				}
																			} else {
																				path.add("Standard Deviation Z > 94.08729");
																				path.add("Inactive");
																				return path;
																			}
																		} else {
																			path.add("Entropy Y > 8.748202");
																			if (stdev_heightY <= 12) {
																				path.add("Stdev Height Y <= 12");
																				if (stdev_valleysX <= 29.933259) {
																					path.add("Standard Deviation Valleys X <= 29.933259");
																					path.add("Washing Hands");
																					return path;
																				} else {
																					path.add("Standard Deviation Valleys X > 29.933259");
																					path.add("Inactive");
																					return path;
																				}
																			} else {
																				path.add("Stdev Height Y > 12");
																				path.add("Inactive");
																				return path;
																			}
																		}
																	}
																}
															}
														} else {
															path.add("Axis Order YZ > 0");
															if (avgXZ <= -107.428571) {
																path.add("Average XZ <= -107.428571");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Average XZ > -107.428571");
																path.add("Inactive");
																return path;
															}
														}
													}
												} else {
													path.add("Standard Deviation Valleys Z > 25.095019");
													if (stdev_peaksX <= 28.827071) {
														path.add("Standard Deviation Peaks X <= 28.827071");
														if (entropyZ <= 5.245837) {
															path.add("Entropy Z <= 5.245837");
															if (avg_jerkZ <= 81.87237) {
																path.add("Avg Jerk Z <= 81.87237");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Avg Jerk Z > 81.87237");
																path.add("Inactive");
																return path;
															}
														} else {
															path.add("Entropy Z > 5.245837");
															path.add("Inactive");
															return path;
														}
													} else {
														path.add("Standard Deviation Peaks X > 28.827071");
														path.add("Washing Hands");
														return path;
													}
												}
											}
										}
									} else {
										path.add("Average Y > 955.333333");
										if (num_valleysX <= 2) {
											path.add("Num Valleys X <= 2");
											if (entropyX <= 11.477528) {
												path.add("Entropy X <= 11.477528");
												if (avg_heightY <= 380) {
													path.add("Avg Height Y <= 380");
													if (num_peaksZ <= 0) {
														path.add("Num Peaks Z <= 0");
														if (corrXZ <= 17144.16327) {
															path.add("Correlation XZ <= 17144.16327");
															path.add("Inactive");
															return path;
														} else {
															path.add("Correlation XZ > 17144.16327");
															path.add("Washing Hands");
															return path;
														}
													} else {
														path.add("Num Peaks Z > 0");
														if (avg_valleysY <= 972) {
															path.add("Average Valleys Y <= 972");
															path.add("Washing Hands");
															return path;
														} else {
															path.add("Average Valleys Y > 972");
															if (avgY <= 1005.333333) {
																path.add("Average Y <= 1005.333333");
																path.add("Inactive");
																return path;
															} else {
																path.add("Average Y > 1005.333333");
																path.add("Washing Hands");
																return path;
															}
														}
													}
												} else {
													path.add("Avg Height Y > 380");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Entropy X > 11.477528");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("Num Valleys X > 2");
											path.add("Washing Hands");
											return path;
										}
									}
								}
							}
						}
					} else {
						path.add("Standard Deviation Z > 379.821559");
						if (corrXY <= -303941.3333) {
							path.add("Correlation XY <= -303941.3333");
							if (avg_peaksZ <= 628) {
								path.add("Average Peaks Z <= 628");
								if (avgX <= -797.767442) {
									path.add("Average X <= -797.767442");
									path.add("Inactive");
									return path;
								} else {
									path.add("Average X > -797.767442");
									if (corrXY <= -392134.5306) {
										path.add("Correlation XY <= -392134.5306");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Correlation XY > -392134.5306");
										if (stdevZ <= 424.458182) {
											path.add("Standard Deviation Z <= 424.458182");
											path.add("Inactive");
											return path;
										} else {
											path.add("Standard Deviation Z > 424.458182");
											if (avg_heightZ <= 998) {
												path.add("Avg Height Z <= 998");
												path.add("Washing Hands");
												return path;
											} else {
												path.add("Avg Height Z > 998");
												path.add("Inactive");
												return path;
											}
										}
									}
								}
							} else {
								path.add("Average Peaks Z > 628");
								path.add("Inactive");
								return path;
							}
						} else {
							path.add("Correlation XY > -303941.3333");
							if (rmsY <= 917.693485) {
								path.add("RMS Y <= 917.693485");
								if (corrXY <= 11190.26087) {
									path.add("Correlation XY <= 11190.26087");
									if (avg_peaksX <= 333.333333) {
										path.add("Average Peaks X <= 333.333333");
										path.add("Inactive");
										return path;
									} else {
										path.add("Average Peaks X > 333.333333");
										if (num_valleysY <= 1) {
											path.add("Num Valleys Y <= 1");
											if (stdev_heightZ <= 118.118584) {
												path.add("Stdev Height Z <= 118.118584");
												if (avgX <= -43.744681) {
													path.add("Average X <= -43.744681");
													if (energyY <= 28600000) {
														path.add("Energy Y <= 28600000");
														path.add("Washing Hands");
														return path;
													} else {
														path.add("Energy Y > 28600000");
														path.add("Inactive");
														return path;
													}
												} else {
													path.add("Average X > -43.744681");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Stdev Height Z > 118.118584");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Num Valleys Y > 1");
											path.add("Washing Hands");
											return path;
										}
									}
								} else {
									path.add("Correlation XY > 11190.26087");
									path.add("Inactive");
									return path;
								}
							} else {
								path.add("RMS Y > 917.693485");
								if (axis_orderXZ <= 27) {
									path.add("Axis Order XZ <= 27");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Axis Order XZ > 27");
									path.add("Inactive");
									return path;
								}
							}
						}
					}
				} else {
					path.add("Average Valleys X > -304");
					if (avgXY <= -1124.333333) {
						path.add("Average XY <= -1124.333333");
						if (avgXZ <= 34.56) {
							path.add("Average XZ <= 34.56");
							if (avgZ <= -167.836735) {
								path.add("Average Z <= -167.836735");
								path.add("Washing Hands");
								return path;
							} else {
								path.add("Average Z > -167.836735");
								path.add("Inactive");
								return path;
							}
						} else {
							path.add("Average XZ > 34.56");
							path.add("Inactive");
							return path;
						}
					} else {
						path.add("Average XY > -1124.333333");
						path.add("Inactive");
						return path;
					}
				}
			} else {
				path.add("Average YZ > 5028");
				path.add("Washing Hands");
				return path;
			}
		} else {
			path.add("Average Peaks Z > 669.333333");
			if (avgX <= -341.714286) {
				path.add("Average X <= -341.714286");
				if (entropyY <= 3.109854) {
					path.add("Entropy Y <= 3.109854");
					if (energyZ <= 31100000) {
						path.add("Energy Z <= 31100000");
						path.add("Inactive");
						return path;
					} else {
						path.add("Energy Z > 31100000");
						path.add("Washing Hands");
						return path;
					}
				} else {
					path.add("Entropy Y > 3.109854");
					if (avgY <= 365.617021) {
						path.add("Average Y <= 365.617021");
						path.add("Inactive");
						return path;
					} else {
						path.add("Average Y > 365.617021");
						if (energyZ <= 8618240) {
							path.add("Energy Z <= 8618240");
							if (num_peaksX <= 1) {
								path.add("Num Peaks X <= 1");
								if (entropyX <= 3.882835) {
									path.add("Entropy X <= 3.882835");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Entropy X > 3.882835");
									path.add("Inactive");
									return path;
								}
							} else {
								path.add("Num Peaks X > 1");
								path.add("Washing Hands");
								return path;
							}
						} else {
							path.add("Energy Z > 8618240");
							if (avg_valleysX <= -562.666667) {
								path.add("Average Valleys X <= -562.666667");
								if (avgY <= 489.565217) {
									path.add("Average Y <= 489.565217");
									if (stdevX <= 137.550667) {
										path.add("Standard Deviation X <= 137.550667");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Standard Deviation X > 137.550667");
										path.add("Inactive");
										return path;
									}
								} else {
									path.add("Average Y > 489.565217");
									if (corrXY <= -238144) {
										path.add("Correlation XY <= -238144");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Correlation XY > -238144");
										if (stdev_peaksY <= 22.939534) {
											path.add("Standard Deviation Peaks Y <= 22.939534");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Standard Deviation Peaks Y > 22.939534");
											path.add("Inactive");
											return path;
										}
									}
								}
							} else {
								path.add("Average Valleys X > -562.666667");
								if (corrYZ <= 441567.2889) {
									path.add("Correlation YZ <= 441567.2889");
									if (num_peaksY <= 0) {
										path.add("Num Peaks Y <= 0");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Num Peaks Y > 0");
										if (num_valleysZ <= 0) {
											path.add("Num Valleys Z <= 0");
											path.add("Inactive");
											return path;
										} else {
											path.add("Num Valleys Z > 0");
											if (avg_heightY <= 530) {
												path.add("Avg Height Y <= 530");
												if (avg_heightZ <= 617.333333) {
													path.add("Avg Height Z <= 617.333333");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Avg Height Z > 617.333333");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Avg Height Y > 530");
												path.add("Washing Hands");
												return path;
											}
										}
									}
								} else {
									path.add("Correlation YZ > 441567.2889");
									path.add("Inactive");
									return path;
								}
							}
						}
					}
				}
			} else {
				path.add("Average X > -341.714286");
				if (avgYZ <= 1182.5) {
					path.add("Average YZ <= 1182.5");
					if (avgXY <= -119.333333) {
						path.add("Average XY <= -119.333333");
						if (avg_peaksZ <= 850) {
							path.add("Average Peaks Z <= 850");
							if (num_peaksZ <= 0) {
								path.add("Num Peaks Z <= 0");
								if (avg_valleysZ <= -728) {
									path.add("Average Valleys Z <= -728");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Average Valleys Z > -728");
									if (axis_orderYZ <= 32) {
										path.add("Axis Order YZ <= 32");
										path.add("Inactive");
										return path;
									} else {
										path.add("Axis Order YZ > 32");
										if (avg_jerkZ <= 191.50945) {
											path.add("Avg Jerk Z <= 191.50945");
											path.add("Inactive");
											return path;
										} else {
											path.add("Avg Jerk Z > 191.50945");
											path.add("Washing Hands");
											return path;
										}
									}
								}
							} else {
								path.add("Num Peaks Z > 0");
								if (avg_heightZ <= 1637.333333) {
									path.add("Avg Height Z <= 1637.333333");
									path.add("Inactive");
									return path;
								} else {
									path.add("Avg Height Z > 1637.333333");
									if (axis_orderXZ <= 17) {
										path.add("Axis Order XZ <= 17");
										if (avgX <= 254.901961) {
											path.add("Average X <= 254.901961");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Average X > 254.901961");
											path.add("Inactive");
											return path;
										}
									} else {
										path.add("Axis Order XZ > 17");
										path.add("Inactive");
										return path;
									}
								}
							}
						} else {
							path.add("Average Peaks Z > 850");
							if (entropyX <= 14.405422) {
								path.add("Entropy X <= 14.405422");
								if (corrXY <= -80305.63265) {
									path.add("Correlation XY <= -80305.63265");
									if (stdev_valleysX <= 63.231497) {
										path.add("Standard Deviation Valleys X <= 63.231497");
										if (avg_peaksZ <= 868) {
											path.add("Average Peaks Z <= 868");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Average Peaks Z > 868");
											if (axis_orderXY <= 3) {
												path.add("Axis Order XY <= 3");
												if (energyZ <= 38200000) {
													path.add("Energy Z <= 38200000");
													if (stdev_peaksX <= 30.942527) {
														path.add("Standard Deviation Peaks X <= 30.942527");
														if (entropyZ <= 13.42003) {
															path.add("Entropy Z <= 13.42003");
															if (axis_orderYZ <= 13) {
																path.add("Axis Order YZ <= 13");
																path.add("Inactive");
																return path;
															} else {
																path.add("Axis Order YZ > 13");
																if (avg_jerkX <= 106.85332) {
																	path.add("Avg Jerk X <= 106.85332");
																	if (avg_valleysY <= 612) {
																		path.add("Average Valleys Y <= 612");
																		if (num_peaksZ <= 1) {
																			path.add("Num Peaks Z <= 1");
																			if (axis_orderYZ <= 25) {
																				path.add("Axis Order YZ <= 25");
																				if (avgYZ <= 140.96) {
																					path.add("Average YZ <= 140.96");
																					path.add("Washing Hands");
																					return path;
																				} else {
																					path.add("Average YZ > 140.96");
																					path.add("Inactive");
																					return path;
																				}
																			} else {
																				path.add("Axis Order YZ > 25");
																				path.add("Inactive");
																				return path;
																			}
																		} else {
																			path.add("Num Peaks Z > 1");
																			path.add("Inactive");
																			return path;
																		}
																	} else {
																		path.add("Average Valleys Y > 612");
																		path.add("Washing Hands");
																		return path;
																	}
																} else {
																	path.add("Avg Jerk X > 106.85332");
																	path.add("Washing Hands");
																	return path;
																}
															}
														} else {
															path.add("Entropy Z > 13.42003");
															if (avgY <= 934.608696) {
																path.add("Average Y <= 934.608696");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Average Y > 934.608696");
																if (num_peaksY <= 0) {
																	path.add("Num Peaks Y <= 0");
																	path.add("Washing Hands");
																	return path;
																} else {
																	path.add("Num Peaks Y > 0");
																	path.add("Inactive");
																	return path;
																}
															}
														}
													} else {
														path.add("Standard Deviation Peaks X > 30.942527");
														path.add("Washing Hands");
														return path;
													}
												} else {
													path.add("Energy Z > 38200000");
													if (stdev_valleysZ <= 38.688711) {
														path.add("Standard Deviation Valleys Z <= 38.688711");
														path.add("Washing Hands");
														return path;
													} else {
														path.add("Standard Deviation Valleys Z > 38.688711");
														path.add("Inactive");
														return path;
													}
												}
											} else {
												path.add("Axis Order XY > 3");
												path.add("Inactive");
												return path;
											}
										}
									} else {
										path.add("Standard Deviation Valleys X > 63.231497");
										path.add("Washing Hands");
										return path;
									}
								} else {
									path.add("Correlation XY > -80305.63265");
									if (num_valleysZ <= 1) {
										path.add("Num Valleys Z <= 1");
										if (avg_heightZ <= 2128) {
											path.add("Avg Height Z <= 2128");
											if (avg_peaksZ <= 932) {
												path.add("Average Peaks Z <= 932");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average Peaks Z > 932");
												if (avg_peaksZ <= 963.428571) {
													path.add("Average Peaks Z <= 963.428571");
													if (num_peaksZ <= 0) {
														path.add("Num Peaks Z <= 0");
														path.add("Washing Hands");
														return path;
													} else {
														path.add("Num Peaks Z > 0");
														if (avgX <= 199.130435) {
															path.add("Average X <= 199.130435");
															path.add("Inactive");
															return path;
														} else {
															path.add("Average X > 199.130435");
															path.add("Washing Hands");
															return path;
														}
													}
												} else {
													path.add("Average Peaks Z > 963.428571");
													if (energyZ <= 14700000) {
														path.add("Energy Z <= 14700000");
														path.add("Inactive");
														return path;
													} else {
														path.add("Energy Z > 14700000");
														if (stdev_peaksY <= 48.106548) {
															path.add("Standard Deviation Peaks Y <= 48.106548");
															if (num_valleysY <= 0) {
																path.add("Num Valleys Y <= 0");
																if (avg_jerkZ <= -215.04065) {
																	path.add("Avg Jerk Z <= -215.04065");
																	path.add("Inactive");
																	return path;
																} else {
																	path.add("Avg Jerk Z > -215.04065");
																	if (avgX <= -139.833333) {
																		path.add("Average X <= -139.833333");
																		path.add("Inactive");
																		return path;
																	} else {
																		path.add("Average X > -139.833333");
																		path.add("Washing Hands");
																		return path;
																	}
																}
															} else {
																path.add("Num Valleys Y > 0");
																if (avg_peaksZ <= 1092) {
																	path.add("Average Peaks Z <= 1092");
																	if (avg_peaksY <= 613.333333) {
																		path.add("Average Peaks Y <= 613.333333");
																		path.add("Washing Hands");
																		return path;
																	} else {
																		path.add("Average Peaks Y > 613.333333");
																		path.add("Inactive");
																		return path;
																	}
																} else {
																	path.add("Average Peaks Z > 1092");
																	path.add("Inactive");
																	return path;
																}
															}
														} else {
															path.add("Standard Deviation Peaks Y > 48.106548");
															path.add("Washing Hands");
															return path;
														}
													}
												}
											}
										} else {
											path.add("Avg Height Z > 2128");
											if (avg_peaksZ <= 1168) {
												path.add("Average Peaks Z <= 1168");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average Peaks Z > 1168");
												if (avg_jerkX <= 311.25802) {
													path.add("Avg Jerk X <= 311.25802");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Avg Jerk X > 311.25802");
													path.add("Inactive");
													return path;
												}
											}
										}
									} else {
										path.add("Num Valleys Z > 1");
										if (avgXY <= -841.44) {
											path.add("Average XY <= -841.44");
											if (energyY <= 39000000) {
												path.add("Energy Y <= 39000000");
												path.add("Inactive");
												return path;
											} else {
												path.add("Energy Y > 39000000");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Average XY > -841.44");
											path.add("Inactive");
											return path;
										}
									}
								}
							} else {
								path.add("Entropy X > 14.405422");
								if (avg_jerkY <= 146.08174) {
									path.add("Avg Jerk Y <= 146.08174");
									if (axis_orderXY <= 5) {
										path.add("Axis Order XY <= 5");
										if (axis_orderXY <= 0) {
											path.add("Axis Order XY <= 0");
											if (stdev_peaksY <= 62.833112) {
												path.add("Standard Deviation Peaks Y <= 62.833112");
												if (avgXY <= -699.2) {
													path.add("Average XY <= -699.2");
													path.add("Inactive");
													return path;
												} else {
													path.add("Average XY > -699.2");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Standard Deviation Peaks Y > 62.833112");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Axis Order XY > 0");
											path.add("Washing Hands");
											return path;
										}
									} else {
										path.add("Axis Order XY > 5");
										path.add("Inactive");
										return path;
									}
								} else {
									path.add("Avg Jerk Y > 146.08174");
									path.add("Washing Hands");
									return path;
								}
							}
						}
					} else {
						path.add("Average XY > -119.333333");
						path.add("Inactive");
						return path;
					}
				} else {
					path.add("Average YZ > 1182.5");
					if (stdevY <= 130.96522) {
						path.add("Standard Deviation Y <= 130.96522");
						path.add("Inactive");
						return path;
					} else {
						path.add("Standard Deviation Y > 130.96522");
						if (num_valleysZ <= 2) {
							path.add("Num Valleys Z <= 2");
							if (avgXY <= -768.32) {
								path.add("Average XY <= -768.32");
								path.add("Washing Hands");
								return path;
							} else {
								path.add("Average XY > -768.32");
								if (entropyX <= 11.945926) {
									path.add("Entropy X <= 11.945926");
									path.add("Inactive");
									return path;
								} else {
									path.add("Entropy X > 11.945926");
									path.add("Washing Hands");
									return path;
								}
							}
						} else {
							path.add("Num Valleys Z > 2");
							path.add("Inactive");
							return path;
						}
					}
				}
			}
		}
	}
	public static ArrayList<String> j48_washhands (String[] line, String filename) {
		Double avg_jerkX = Double.parseDouble(line[0]);
		Double avg_jerkY = Double.parseDouble(line[1]);
		Double avg_jerkZ = Double.parseDouble(line[2]);
		Double avg_heightX = Double.parseDouble(line[3]);
		Double avg_heightY = Double.parseDouble(line[4]);
		Double avg_heightZ = Double.parseDouble(line[5]);
		Double stdev_heightX = Double.parseDouble(line[6]);
		Double stdev_heightY = Double.parseDouble(line[7]);
		Double stdev_heightZ = Double.parseDouble(line[8]);
		Double avg_dist_meanX = Double.parseDouble(line[10]);
		Double avg_dist_meanY = Double.parseDouble(line[11]);
		Double avg_dist_meanZ = Double.parseDouble(line[12]);
		Double stdev_meanX = Double.parseDouble(line[13]);
		Double stdev_meanY = Double.parseDouble(line[14]);
		Double stdev_meanZ = Double.parseDouble(line[15]);
		Double energyX = Double.parseDouble(line[9]);
		Double energyY = Double.parseDouble(line[10]);
		Double energyZ = Double.parseDouble(line[11]);
		Double entropyX = Double.parseDouble(line[12]);
		Double entropyY = Double.parseDouble(line[13]);
		Double entropyZ = Double.parseDouble(line[14]);
		Double avgX = Double.parseDouble(line[15]);
		Double avgY = Double.parseDouble(line[16]);
		Double avgZ = Double.parseDouble(line[17]);
		Double avgXY = Double.parseDouble(line[18]);
		Double avgXZ = Double.parseDouble(line[19]);
		Double avgYZ = Double.parseDouble(line[20]);
		Double stdevX = Double.parseDouble(line[21]);
		Double stdevY = Double.parseDouble(line[22]);
		Double stdevZ = Double.parseDouble(line[23]);
		Double corrXY = Double.parseDouble(line[24]);
		Double corrXZ = Double.parseDouble(line[25]);
		Double corrYZ = Double.parseDouble(line[26]);
		Double rmsX = Double.parseDouble(line[27]);
		Double rmsY = Double.parseDouble(line[28]);
		Double rmsZ = Double.parseDouble(line[29]);
		Double axis_orderXY = Double.parseDouble(line[30]);
		Double axis_orderXZ = Double.parseDouble(line[31]);
		Double axis_orderYZ = Double.parseDouble(line[32]);
		Double num_peaksX = Double.parseDouble(line[33]);
		Double num_peaksY = Double.parseDouble(line[34]);
		Double num_peaksZ = Double.parseDouble(line[35]);
		Double avg_peaksX = Double.parseDouble(line[36]);
		Double avg_peaksY = Double.parseDouble(line[37]);
		Double avg_peaksZ = Double.parseDouble(line[38]);
		Double stdev_peaksX = Double.parseDouble(line[39]);
		Double stdev_peaksY = Double.parseDouble(line[40]);
		Double stdev_peaksZ = Double.parseDouble(line[41]);
		Double num_valleysX = Double.parseDouble(line[42]);
		Double num_valleysY = Double.parseDouble(line[43]);
		Double num_valleysZ = Double.parseDouble(line[44]);
		Double avg_valleysX = Double.parseDouble(line[45]);
		Double avg_valleysY = Double.parseDouble(line[46]);
		Double avg_valleysZ = Double.parseDouble(line[47]);
		Double stdev_valleysX = Double.parseDouble(line[48]);
		Double stdev_valleysY = Double.parseDouble(line[49]);
		Double stdev_valleysZ = Double.parseDouble(line[50]);
		//Double zero_crossingsX = Double.parseDouble(line[51]);
		//Double zero_crossingsY = Double.parseDouble(line[52]);
		//Double zero_crossingsZ = Double.parseDouble(line[53]);
		//Double num_points = Double.parseDouble(line[54]);
		//String gesture = line[51];
		//Double start_time = Double.parseDouble(line[52]);
		//Double end_time = Double.parseDouble(line[53]);
		ArrayList<String> path = new ArrayList<String>();
		if (avg_heightZ <= 1620.923077) {
			path.add("Avg Height Z <= 1620.923077");
			if (avg_peaksZ <= 658.666667) {
				path.add("Average Peaks Z <= 658.666667");
				if (avgYZ <= 1609.73913) {
					path.add("Average YZ <= 1609.73913");
					if (avg_jerkZ <= 153.17431) {
						path.add("Avg Jerk Z <= 153.17431");
						if (avgXY <= -971.878788) {
							path.add("Average XY <= -971.878788");
							if (stdev_heightX <= 136.235091) {
								path.add("Stdev Height X <= 136.235091");
								if (stdevZ <= 304.507424) {
									path.add("Standard Deviation Z <= 304.507424");
									if (corrXY <= -394863.831579) {
										path.add("Correlation XY <= -394863.831579");
										if (rmsZ <= 48.325427) {
											path.add("RMS Z <= 48.325427");
											if (avgXY <= -1400.727273) {
												path.add("Average XY <= -1400.727273");
												if (stdev_valleysY <= 4) {
													path.add("Standard Deviation Valleys Y <= 4");
													path.add("Inactive");
													return path;
												} else {
													path.add("Standard Deviation Valleys Y > 4");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Average XY > -1400.727273");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("RMS Z > 48.325427");
											if (rmsX <= 524.304081) {
												path.add("RMS X <= 524.304081");
												path.add("Inactive");
												return path;
											} else {
												path.add("RMS X > 524.304081");
												if (avgX <= -817.11828) {
													path.add("Average X <= -817.11828");
													path.add("Inactive");
													return path;
												} else {
													path.add("Average X > -817.11828");
													if (avgY <= 551.083333) {
														path.add("Average Y <= 551.083333");
														if (entropyX <= 5.170908) {
															path.add("Entropy X <= 5.170908");
															path.add("Washing Hands");
															return path;
														} else {
															path.add("Entropy X > 5.170908");
															path.add("Inactive");
															return path;
														}
													} else {
														path.add("Average Y > 551.083333");
														if (num_valleysX <= 0) {
															path.add("Num Valleys X <= 0");
															if (axis_orderXY <= 0) {
																path.add("Axis Order XY <= 0");
																if (avg_jerkZ <= -64.86938) {
																	path.add("Avg Jerk Z <= -64.86938");
																	path.add("Washing Hands");
																	return path;
																} else {
																	path.add("Avg Jerk Z > -64.86938");
																	path.add("Inactive");
																	return path;
																}
															} else {
																path.add("Axis Order XY > 0");
																path.add("Washing Hands");
																return path;
															}
														} else {
															path.add("Num Valleys X > 0");
															path.add("Inactive");
															return path;
														}
													}
												}
											}
										}
									} else {
										path.add("Correlation XY > -394863.831579");
										if (rmsZ <= 206.708708) {
											path.add("RMS Z <= 206.708708");
											if (corrYZ <= -138175.326316) {
												path.add("Correlation YZ <= -138175.326316");
												if (axis_orderXZ <= 10) {
													path.add("Axis Order XZ <= 10");
													if (stdev_valleysZ <= 11.313709) {
														path.add("Standard Deviation Valleys Z <= 11.313709");
														path.add("Inactive");
														return path;
													} else {
														path.add("Standard Deviation Valleys Z > 11.313709");
														if (energyY <= 98455680) {
															path.add("Energy Y <= 98455680");
															path.add("Washing Hands");
															return path;
														} else {
															path.add("Energy Y > 98455680");
															path.add("Inactive");
															return path;
														}
													}
												} else {
													path.add("Axis Order XZ > 10");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Correlation YZ > -138175.326316");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("RMS Z > 206.708708");
											if (avgXZ <= 325.938144) {
												path.add("Average XZ <= 325.938144");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average XZ > 325.938144");
												if (stdevZ <= 24.580917) {
													path.add("Standard Deviation Z <= 24.580917");
													if (avg_valleysX <= -264) {
														path.add("Average Valleys X <= -264");
														path.add("Inactive");
														return path;
													} else {
														path.add("Average Valleys X > -264");
														path.add("Washing Hands");
														return path;
													}
												} else {
													path.add("Standard Deviation Z > 24.580917");
													path.add("Inactive");
													return path;
												}
											}
										}
									}
								} else {
									path.add("Standard Deviation Z > 304.507424");
									if (avgXZ <= -766.583333) {
										path.add("Average XZ <= -766.583333");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Average XZ > -766.583333");
										if (corrXY <= -390405.818182) {
											path.add("Correlation XY <= -390405.818182");
											if (num_peaksX <= 2) {
												path.add("Num Peaks X <= 2");
												if (entropyZ <= 14.950075) {
													path.add("Entropy Z <= 14.950075");
													path.add("Inactive");
													return path;
												} else {
													path.add("Entropy Z > 14.950075");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Num Peaks X > 2");
												path.add("Inactive");
												return path;
											}
										} else {
											path.add("Correlation XY > -390405.818182");
											if (entropyZ <= 12.024043) {
												path.add("Entropy Z <= 12.024043");
												if (avg_valleysX <= -728) {
													path.add("Average Valleys X <= -728");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Average Valleys X > -728");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Entropy Z > 12.024043");
												if (num_peaksX <= 2) {
													path.add("Num Peaks X <= 2");
													if (axis_orderYZ <= 6) {
														path.add("Axis Order YZ <= 6");
														path.add("Inactive");
														return path;
													} else {
														path.add("Axis Order YZ > 6");
														if (avg_valleysY <= 216) {
															path.add("Average Valleys Y <= 216");
															path.add("Inactive");
															return path;
														} else {
															path.add("Average Valleys Y > 216");
															path.add("Washing Hands");
															return path;
														}
													}
												} else {
													path.add("Num Peaks X > 2");
													if (avg_jerkZ <= -80.88364) {
														path.add("Avg Jerk Z <= -80.88364");
														path.add("Washing Hands");
														return path;
													} else {
														path.add("Avg Jerk Z > -80.88364");
														path.add("Inactive");
														return path;
													}
												}
											}
										}
									}
								}
							} else {
								path.add("Stdev Height X > 136.235091");
								if (avg_valleysZ <= -948) {
									path.add("Average Valleys Z <= -948");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Average Valleys Z > -948");
									if (num_valleysX <= 2) {
										path.add("Num Valleys X <= 2");
										path.add("Inactive");
										return path;
									} else {
										path.add("Num Valleys X > 2");
										if (corrXY <= -471721.094737) {
											path.add("Correlation XY <= -471721.094737");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Correlation XY > -471721.094737");
											path.add("Inactive");
											return path;
										}
									}
								}
							}
						} else {
							path.add("Average XY > -971.878788");
							if (avg_peaksX <= 1158) {
								path.add("Average Peaks X <= 1158");
								path.add("Inactive");
								return path;
							} else {
								path.add("Average Peaks X > 1158");
								if (avgXZ <= 429.494949) {
									path.add("Average XZ <= 429.494949");
									if (stdevX <= 417.553083) {
										path.add("Standard Deviation X <= 417.553083");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Standard Deviation X > 417.553083");
										path.add("Inactive");
										return path;
									}
								} else {
									path.add("Average XZ > 429.494949");
									path.add("Inactive");
									return path;
								}
							}
						}
					} else {
						path.add("Avg Jerk Z > 153.17431");
						if (avgX <= -518.905263) {
							path.add("Average X <= -518.905263");
							if (entropyY <= 5.604037) {
								path.add("Entropy Y <= 5.604037");
								path.add("Washing Hands");
								return path;
							} else {
								path.add("Entropy Y > 5.604037");
								if (rmsX <= 677.102714) {
									path.add("RMS X <= 677.102714");
									if (avg_valleysX <= -992) {
										path.add("Average Valleys X <= -992");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Average Valleys X > -992");
										if (num_valleysZ <= 1) {
											path.add("Num Valleys Z <= 1");
											if (avg_valleysY <= 184) {
												path.add("Average Valleys Y <= 184");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average Valleys Y > 184");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Num Valleys Z > 1");
											path.add("Inactive");
											return path;
										}
									}
								} else {
									path.add("RMS X > 677.102714");
									if (num_valleysX <= 1) {
										path.add("Num Valleys X <= 1");
										path.add("Inactive");
										return path;
									} else {
										path.add("Num Valleys X > 1");
										if (avg_jerkY <= 150.0818) {
											path.add("Avg Jerk Y <= 150.0818");
											path.add("Inactive");
											return path;
										} else {
											path.add("Avg Jerk Y > 150.0818");
											path.add("Washing Hands");
											return path;
										}
									}
								}
							}
						} else {
							path.add("Average X > -518.905263");
							if (stdevZ <= 541.585219) {
								path.add("Standard Deviation Z <= 541.585219");
								if (num_valleysX <= 0) {
									path.add("Num Valleys X <= 0");
									if (num_peaksZ <= 0) {
										path.add("Num Peaks Z <= 0");
										if (rmsZ <= 548.346978) {
											path.add("RMS Z <= 548.346978");
											if (avg_peaksZ <= -464) {
												path.add("Average Peaks Z <= -464");
												if (avg_jerkX <= -79.8608) {
													path.add("Avg Jerk X <= -79.8608");
													path.add("Inactive");
													return path;
												} else {
													path.add("Avg Jerk X > -79.8608");
													path.add("Washing Hands");
													return path;
												}
											} else {
												path.add("Average Peaks Z > -464");
												if (avg_valleysZ <= -400) {
													path.add("Average Valleys Z <= -400");
													if (stdevX <= 252.09533) {
														path.add("Standard Deviation X <= 252.09533");
														path.add("Inactive");
														return path;
													} else {
														path.add("Standard Deviation X > 252.09533");
														if (stdevX <= 390.059404) {
															path.add("Standard Deviation X <= 390.059404");
															if (avg_valleysY <= 248) {
																path.add("Average Valleys Y <= 248");
																path.add("Washing Hands");
																return path;
															} else {
																path.add("Average Valleys Y > 248");
																path.add("Inactive");
																return path;
															}
														} else {
															path.add("Standard Deviation X > 390.059404");
															path.add("Inactive");
															return path;
														}
													}
												} else {
													path.add("Average Valleys Z > -400");
													if (stdevX <= 94.71263) {
														path.add("Standard Deviation X <= 94.71263");
														path.add("Inactive");
														return path;
													} else {
														path.add("Standard Deviation X > 94.71263");
														path.add("Washing Hands");
														return path;
													}
												}
											}
										} else {
											path.add("RMS Z > 548.346978");
											path.add("Inactive");
											return path;
										}
									} else {
										path.add("Num Peaks Z > 0");
										path.add("Inactive");
										return path;
									}
								} else {
									path.add("Num Valleys X > 0");
									path.add("Inactive");
									return path;
								}
							} else {
								path.add("Standard Deviation Z > 541.585219");
								if (avg_peaksX <= -308) {
									path.add("Average Peaks X <= -308");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Average Peaks X > -308");
									if (num_peaksX <= 0) {
										path.add("Num Peaks X <= 0");
										path.add("Inactive");
										return path;
									} else {
										path.add("Num Peaks X > 0");
										if (stdevY <= 490.621402) {
											path.add("Standard Deviation Y <= 490.621402");
											if (stdev_peaksY <= 34.161382) {
												path.add("Standard Deviation Peaks Y <= 34.161382");
												if (avg_valleysZ <= -936) {
													path.add("Average Valleys Z <= -936");
													path.add("Inactive");
													return path;
												} else {
													path.add("Average Valleys Z > -936");
													if (axis_orderXY <= 19) {
														path.add("Axis Order XY <= 19");
														path.add("Inactive");
														return path;
													} else {
														path.add("Axis Order XY > 19");
														path.add("Washing Hands");
														return path;
													}
												}
											} else {
												path.add("Standard Deviation Peaks Y > 34.161382");
												if (energyY <= 60512000) {
													path.add("Energy Y <= 60512000");
													path.add("Inactive");
													return path;
												} else {
													path.add("Energy Y > 60512000");
													path.add("Washing Hands");
													return path;
												}
											}
										} else {
											path.add("Standard Deviation Y > 490.621402");
											if (avg_peaksX <= 684) {
												path.add("Average Peaks X <= 684");
												path.add("Washing Hands");
												return path;
											} else {
												path.add("Average Peaks X > 684");
												path.add("Inactive");
												return path;
											}
										}
									}
								}
							}
						}
					}
				} else {
					path.add("Average YZ > 1609.73913");
					if (avgXY <= -17980) {
						path.add("Average XY <= -17980");
						path.add("Washing Hands");
						return path;
					} else {
						path.add("Average XY > -17980");
						path.add("Inactive");
						return path;
					}
				}
			} else {
				path.add("Average Peaks Z > 658.666667");
				if (avgX <= -331.784946) {
					path.add("Average X <= -331.784946");
					if (stdevX <= 353.743368) {
						path.add("Standard Deviation X <= 353.743368");
						if (corrYZ <= 420206.6087) {
							path.add("Correlation YZ <= 420206.6087");
							if (avgY <= 392.989691) {
								path.add("Average Y <= 392.989691");
								path.add("Inactive");
								return path;
							} else {
								path.add("Average Y > 392.989691");
								if (avg_valleysY <= 28) {
									path.add("Average Valleys Y <= 28");
									if (avg_heightZ <= 645.333333) {
										path.add("Avg Height Z <= 645.333333");
										path.add("Inactive");
										return path;
									} else {
										path.add("Avg Height Z > 645.333333");
										path.add("Washing Hands");
										return path;
									}
								} else {
									path.add("Average Valleys Y > 28");
									if (energyZ <= 12900000) {
										path.add("Energy Z <= 12900000");
										if (stdev_heightX <= 9.991997) {
											path.add("Stdev Height X <= 9.991997");
											path.add("Inactive");
											return path;
										} else {
											path.add("Stdev Height X > 9.991997");
											path.add("Washing Hands");
											return path;
										}
									} else {
										path.add("Energy Z > 12900000");
										if (avg_valleysY <= 414) {
											path.add("Average Valleys Y <= 414");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Average Valleys Y > 414");
											if (num_valleysX <= 0) {
												path.add("Num Valleys X <= 0");
												path.add("Inactive");
												return path;
											} else {
												path.add("Num Valleys X > 0");
												if (avg_valleysZ <= 493.333333) {
													path.add("Average Valleys Z <= 493.333333");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Average Valleys Z > 493.333333");
													path.add("Inactive");
													return path;
												}
											}
										}
									}
								}
							}
						} else {
							path.add("Correlation YZ > 420206.6087");
							path.add("Inactive");
							return path;
						}
					} else {
						path.add("Standard Deviation X > 353.743368");
						path.add("Inactive");
						return path;
					}
				} else {
					path.add("Average X > -331.784946");
					if (avg_peaksZ <= 957.333333) {
						path.add("Average Peaks Z <= 957.333333");
						if (num_peaksZ <= 0) {
							path.add("Num Peaks Z <= 0");
							if (avg_peaksZ <= 888) {
								path.add("Average Peaks Z <= 888");
								if (avg_valleysZ <= -800) {
									path.add("Average Valleys Z <= -800");
									if (avg_jerkX <= 49.52921) {
										path.add("Avg Jerk X <= 49.52921");
										path.add("Washing Hands");
										return path;
									} else {
										path.add("Avg Jerk X > 49.52921");
										path.add("Inactive");
										return path;
									}
								} else {
									path.add("Average Valleys Z > -800");
									path.add("Inactive");
									return path;
								}
							} else {
								path.add("Average Peaks Z > 888");
								path.add("Washing Hands");
								return path;
							}
						} else {
							path.add("Num Peaks Z > 0");
							path.add("Inactive");
							return path;
						}
					} else {
						path.add("Average Peaks Z > 957.333333");
						if (num_peaksZ <= 3) {
							path.add("Num Peaks Z <= 3");
							if (num_peaksY <= 2) {
								path.add("Num Peaks Y <= 2");
								if (rmsZ <= 562.293783) {
									path.add("RMS Z <= 562.293783");
									path.add("Inactive");
									return path;
								} else {
									path.add("RMS Z > 562.293783");
									if (num_valleysX <= 2) {
										path.add("Num Valleys X <= 2");
										if (avg_peaksX <= -108) {
											path.add("Average Peaks X <= -108");
											if (avg_peaksX <= -208) {
												path.add("Average Peaks X <= -208");
												path.add("Inactive");
												return path;
											} else {
												path.add("Average Peaks X > -208");
												path.add("Washing Hands");
												return path;
											}
										} else {
											path.add("Average Peaks X > -108");
											if (num_peaksZ <= 2) {
												path.add("Num Peaks Z <= 2");
												if (num_peaksY <= 1) {
													path.add("Num Peaks Y <= 1");
													if (num_peaksZ <= 1) {
														path.add("Num Peaks Z <= 1");
														if (avg_peaksY <= 1408) {
															path.add("Average Peaks Y <= 1408");
															if (avg_jerkX <= -0.97642) {
																path.add("Avg Jerk X <= -0.97642");
																if (corrXY <= -90042.77551) {
																	path.add("Correlation XY <= -90042.77551");
																	path.add("Inactive");
																	return path;
																} else {
																	path.add("Correlation XY > -90042.77551");
																	if (avg_jerkY <= -44.87487) {
																		path.add("Avg Jerk Y <= -44.87487");
																		path.add("Washing Hands");
																		return path;
																	} else {
																		path.add("Avg Jerk Y > -44.87487");
																		if (avg_peaksZ <= 976) {
																			path.add("Average Peaks Z <= 976");
																			path.add("Washing Hands");
																			return path;
																		} else {
																			path.add("Average Peaks Z > 976");
																			if (avg_heightZ <= 1016) {
																				path.add("Avg Height Z <= 1016");
																				path.add("Inactive");
																				return path;
																			} else {
																				path.add("Avg Height Z > 1016");
																				path.add("Washing Hands");
																				return path;
																			}
																		}
																	}
																}
															} else {
																path.add("Avg Jerk X > -0.97642");
																path.add("Inactive");
																return path;
															}
														} else {
															path.add("Average Peaks Y > 1408");
															path.add("Washing Hands");
															return path;
														}
													} else {
														path.add("Num Peaks Z > 1");
														if (stdev_peaksX <= 48) {
															path.add("Standard Deviation Peaks X <= 48");
															if (avg_jerkX <= 4.9943) {
																path.add("Avg Jerk X <= 4.9943");
																path.add("Inactive");
																return path;
															} else {
																path.add("Avg Jerk X > 4.9943");
																path.add("Washing Hands");
																return path;
															}
														} else {
															path.add("Standard Deviation Peaks X > 48");
															path.add("Washing Hands");
															return path;
														}
													}
												} else {
													path.add("Num Peaks Y > 1");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Num Peaks Z > 2");
												path.add("Inactive");
												return path;
											}
										}
									} else {
										path.add("Num Valleys X > 2");
										if (energyY <= 51230912) {
											path.add("Energy Y <= 51230912");
											path.add("Inactive");
											return path;
										} else {
											path.add("Energy Y > 51230912");
											path.add("Washing Hands");
											return path;
										}
									}
								}
							} else {
								path.add("Num Peaks Y > 2");
								if (axis_orderYZ <= 16) {
									path.add("Axis Order YZ <= 16");
									if (avg_heightY <= 1040) {
										path.add("Avg Height Y <= 1040");
										path.add("Inactive");
										return path;
									} else {
										path.add("Avg Height Y > 1040");
										path.add("Washing Hands");
										return path;
									}
								} else {
									path.add("Axis Order YZ > 16");
									path.add("Washing Hands");
									return path;
								}
							}
						} else {
							path.add("Num Peaks Z > 3");
							path.add("Inactive");
							return path;
						}
					}
				}
			}
		} else {
			path.add("Avg Height Z > 1620.923077");
			if (avgXY <= -1024.255319) {
				path.add("Average XY <= -1024.255319");
				if (avg_peaksZ <= 816) {
					path.add("Average Peaks Z <= 816");
					if (avgXZ <= -100.129032) {
						path.add("Average XZ <= -100.129032");
						path.add("Inactive");
						return path;
					} else {
						path.add("Average XZ > -100.129032");
						path.add("Washing Hands");
						return path;
					}
				} else {
					path.add("Average Peaks Z > 816");
					if (energyZ <= 12300416) {
						path.add("Energy Z <= 12300416");
						path.add("Inactive");
						return path;
					} else {
						path.add("Energy Z > 12300416");
						path.add("Washing Hands");
						return path;
					}
				}
			} else {
				path.add("Average XY > -1024.255319");
				if (avgXY <= -254.454545) {
					path.add("Average XY <= -254.454545");
					if (avgXZ <= -414.042553) {
						path.add("Average XZ <= -414.042553");
						if (num_peaksX <= 5) {
							path.add("Num Peaks X <= 5");
							path.add("Washing Hands");
							return path;
						} else {
							path.add("Num Peaks X > 5");
							path.add("Inactive");
							return path;
						}
					} else {
						path.add("Average XZ > -414.042553");
						if (stdev_valleysY <= 164) {
							path.add("Standard Deviation Valleys Y <= 164");
							if (avg_peaksZ <= 668) {
								path.add("Average Peaks Z <= 668");
								path.add("Inactive");
								return path;
							} else {
								path.add("Average Peaks Z > 668");
								if (energyZ <= 25760000) {
									path.add("Energy Z <= 25760000");
									path.add("Inactive");
									return path;
								} else {
									path.add("Energy Z > 25760000");
									if (stdev_valleysX <= 33.941125) {
										path.add("Standard Deviation Valleys X <= 33.941125");
										if (avg_jerkY <= 244.78566) {
											path.add("Avg Jerk Y <= 244.78566");
											if (num_peaksY <= 1) {
												path.add("Num Peaks Y <= 1");
												if (avg_jerkX <= -139.1448) {
													path.add("Avg Jerk X <= -139.1448");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Avg Jerk X > -139.1448");
													path.add("Inactive");
													return path;
												}
											} else {
												path.add("Num Peaks Y > 1");
												if (avg_peaksX <= 316) {
													path.add("Average Peaks X <= 316");
													path.add("Washing Hands");
													return path;
												} else {
													path.add("Average Peaks X > 316");
													path.add("Inactive");
													return path;
												}
											}
										} else {
											path.add("Avg Jerk Y > 244.78566");
											path.add("Washing Hands");
											return path;
										}
									} else {
										path.add("Standard Deviation Valleys X > 33.941125");
										if (stdevZ <= 631.128531) {
											path.add("Standard Deviation Z <= 631.128531");
											path.add("Washing Hands");
											return path;
										} else {
											path.add("Standard Deviation Z > 631.128531");
											path.add("Inactive");
											return path;
										}
									}
								}
							}
						} else {
							path.add("Standard Deviation Valleys Y > 164");
							if (num_valleysX <= 0) {
								path.add("Num Valleys X <= 0");
								path.add("Inactive");
								return path;
							} else {
								path.add("Num Valleys X > 0");
								if (stdev_valleysY <= 240) {
									path.add("Standard Deviation Valleys Y <= 240");
									path.add("Washing Hands");
									return path;
								} else {
									path.add("Standard Deviation Valleys Y > 240");
									path.add("Inactive");
									return path;
								}
							}
						}
					}
				} else {
					path.add("Average XY > -254.454545");
					path.add("Inactive");
					return path;
				}
			}
		}
	}
}
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

//import org.apache.commons.io.FileUtils;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class gesture_recognition {

	final static String[][] master_gesture_list = {{"Brushing Teeth"}};//,"Comb Hair","Drinking","Scratch Chin","Take Medication","Wash Hands"}};
	final static String raw_test_data = "raw data";
	final static String raw_train_data = "Train Data";
	final static String folder = "data/thesis";
	final static String test_folder = "Test Data/Washing Hands";
	final static String output_folder = "Outputs/Washing Hands";
	final static String stats_folder = "Stats/Washing Hands/stats_washhands";
	final static String false_positives_folder = "False Positives Original RF";
	final static String train_filename = "wash_hands41";

	final static int TRUE_POSITIVES = 0;
	final static int FALSE_POSITIVES = 1;
	final static int TRUE_NEGATIVES = 2;
	final static int FALSE_NEGATIVES = 3;

	final static int ACTIVITY = 0;
	final static int INACTIVE = 1;

	//Tier II Parameters
	final  static int min_gesture_time = 15;
	final static double activity_percentage_threshold = 0.75;
	final static int gesture_end_threshold = 5;
	final static double time_interval = 1.0;

	public static void main(String[] args) {

		/**********************************************************************************
		 * 1. Sort data by first column (time), remove duplicates, replace activity names *
		 **********************************************************************************/
		//clean_data();
		//System.out.println("Done Cleaning Data" + "\n");

		//Call misc functions
		//misc.calculate_average_time();
		//misc.graph_false_positives(new File(folder + "/" + stats_folder));
		//misc.phase_three_performance();
		//misc.first_stage_total_time();
		//misc.raw_data_corrupted();
		//misc.combine_csv(new File(folder + "/overall_train_data41.csv"), new File(folder + "/" + test_folder + "/brushing_teeth_46b95a43-6ded-4d61-a3b1-2d1c7cc56f73_628_edited.csv"));
		//misc.separate_graphs(raw_test_data);
		//extract_gesture_data();

		File output_directory = new File (folder + "/" + output_folder);
		File stats_directory = new File (folder + "/" + stats_folder);

		if (!output_directory.exists()) {
			output_directory.mkdir();
		}
		if (!stats_directory.exists()) {
			stats_directory.mkdir();
		}

		for(int a = 0; a < master_gesture_list.length;a++) {
			String[] gesture_list = master_gesture_list[a];
			String gesture_name = "";
			for (int i = 0; i < gesture_list.length; i++) {
				if (gesture_name == "") {
					gesture_name = gesture_name + gesture_list[i].replaceAll(" ","_").toLowerCase().replaceAll(",","_");
				} else {
					gesture_name = gesture_name + "_" + gesture_list[i].replaceAll(" ","_").toLowerCase().replaceAll(",","_");
				}
			}

			/*************************
			 * 2. Generate Test Data *
			 *************************/

			/*Double[] window = new Double[]{0.0,0.0,0.0}; //total, corrupted, corrupted_gesture
			System.out.println("Creating Test Data");
			for (File test_data_file : new File(raw_test_data).listFiles()) {
				if (test_data_file.getName().endsWith(".csv")) {
					System.out.println("Generating: " + test_data_file.getName());
					String test_data =  folder + "/" + test_folder + "/" + gesture_name + "_" + test_data_file.getName();
					window = extract_features.generate_features(test_data_file, test_data, window);
					clean_file(test_data);
					double count = misc.count_gestures(test_data_file);
					if (count > 0) {
						System.out.println("Time: " + count);
					}
				}
			}*/

			/******************************
			 * 3. Train on 1,2, test on 3 *
			 ******************************/

			/********Training Data********/
			String train_data = folder + "/Train Data/" + train_filename;

			if (!new File(train_data + ".arff").exists()) {
				System.out.println("Creating Training Data");

				Double[] window = new Double[]{0.0,0.0,0.0}; //total, corrupted, corrupted_gesture
				for (File fileEntry : new File(raw_train_data).listFiles()) {
					System.out.println("Analyzing: " + fileEntry.getName());
					window = extract_features.generate_features(fileEntry, train_data + ".csv", window);
					System.out.println(misc.count_gestures(fileEntry));
				}
				
				replace_gesture(new File(train_data + ".csv"), gesture_list);
				csv2arff(train_data + ".csv", train_data + ".arff");
				System.out.println("Training Data Created");
			} else {
				train_data = folder + "/Train Data/" + train_filename;
				System.out.println("Training Data Exists");
			}
			/********Testing Data********/
			File test_data_source = new File(folder + "/" + test_folder);
			HashMap<String, Boolean> train_gestures = null;
			HashMap<String, Boolean> test_gestures = null;
			System.out.println("Analysis");
			for (File test_file : test_data_source.listFiles()) {
				if (test_file.getName().endsWith(".csv")) {
					String test_data = test_file.getAbsolutePath();
					System.out.println("Analyzing: " + test_data);

					String testing = test_data.substring(0,test_data.length()-4) + ".arff";
					String output = folder + "/" + output_folder + "/" + test_file.getName().substring(0, test_file.getName().length()-4) + ".csv";
					String falsepos_path = folder + "/False Positive Paths/" + test_file.getName().substring(0, test_file.getName().length()-4) + "_FP.csv";

					if (!new File(testing).exists()) {
						//remove_filename(test_data);
						File train_filecsv = new File(train_data + ".csv");
						File test_filecsv = new File(test_data);
						replace_gesture(test_filecsv, gesture_list);
						test_gestures = alphabetize(new File(test_data));
						train_gestures = alphabetize(new File(train_data + ".csv"));

						add_missing_data(train_gestures, test_gestures, train_filecsv, test_filecsv);
						csv2arff(test_data, test_data.substring(0,test_data.length()-4) + ".arff");
					}
					Classification.algorithm_test(test_data, output, falsepos_path, gesture_name);
					//Classification.weka_classify(testing, training, output);
					//ArrayList<Double> falsenegs = Classification.accuracy(output);
				}
			}

			System.out.println("\n" + "Results");
			//gesture_time(gesture_name, output_folder, stats_folder);
			//tierII_Analysis(output_folder, stats_folder);

			/*for (File output_fol : new File(folder + "/Outputs").listFiles()) {

				if (output_fol.isDirectory()) {
					System.out.println(output_fol.getName());
					gesture_time(gesture_name, "Outputs/" + output_fol.getName(), "Stats/stats".concat(output_fol.getName().substring(6)));
				}
			}*/
			//Classification.tree_analysis(folder, stats_folder);

			/**********************************************************************************************
			 * 4. Do cross-validation on 1,2 - remove incorrect and use as training data for testing on 3 *
			 **********************************************************************************************/
			//do cross validation, maybe it returns the instances not correct
			//redo testing on 3
			/*String train_data_edited = folder + "/overall_train_data41_edited";
			File tempfile = new File(train_data_edited + ".csv");
			if (tempfile.exists()) {
				tempfile.delete();
			}
			try {
				FileUtils.copyFile(new File(train_data + ".csv"), new File(train_data_edited + ".csv"));
				//csv2arff(train_data + ".csv", train_data + ".arff");

				remove_incorrect(Classification.cross_validation(train_data + ".arff"), train_data_edited + ".csv");
				replace_gesture(new File(train_data_edited + ".csv"), gesture_list);
				csv2arff(train_data_edited + ".csv", train_data_edited + ".arff");
				test_data_source = new File(folder + "/" + test_folder);
				train_gestures = null;
				test_gestures = null;
				csv2arff(train_data_edited + ".csv", train_data_edited + ".arff");
				training = train_data_edited +  ".arff";

				for (File test_file : test_data_source.listFiles()) {
					if (test_file.getName().endsWith(".csv")) {
						String test_data = test_file.getAbsolutePath();
						System.out.println(test_data);

						String testing = test_data.substring(0,test_data.length()-4) + ".arff";
						String output = folder + "/outputcrossval/" + test_file.getName().substring(0, test_file.getName().length()-4) + "_" + gesture_name + ".csv";

						Classification.weka_classify(testing, training, output);

						//ArrayList<Double> falsenegs = Classification.accuracy(output);
					}
				}
			gesture_time("brushing_teeth", "outputcrossval", "statscrossval");

			} catch (IOException e) {
				System.out.println("Couldn't copy file");
				e.printStackTrace();
			}*/

			/*try {
				for (double progressPercentage = 0.0; progressPercentage < 1.0; progressPercentage += 0.01) {
					updateProgress(progressPercentage);
					Thread.sleep(20);

				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}*/
		}
	}

	public static void remove_filename(String csvFile) {
		CSVReader reader = null;
		String[] line = null;
		String[] header = null;
		List<String[]> lines = new ArrayList<String[]>();

		try {
			reader = new CSVReader(new FileReader(csvFile));
			header = reader.readNext();
			if (header[0].matches("File Name")) {
				List<String> list = new ArrayList<String>(Arrays.asList(header));
				list.remove(0);
				header = list.toArray(new String[0]);
				lines.add(header);
				while ((line = reader.readNext()) != null) {
					list = new ArrayList<String>(Arrays.asList(line));
					list.remove(0);
					line = list.toArray(new String[0]);
					lines.add(line);
				}
			} else {
				reader.close();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (header[0].matches("File Name")) {
				try {
					reader.close();
					CSVWriter writer = new CSVWriter(new FileWriter(csvFile));
					Iterator<String[]> it = lines.iterator();
					while (it.hasNext()) {
						writer.writeNext(it.next());;
					}
					writer.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static void clean_file (String csvFile) {
		//remove duplicates
		BufferedReader br = null;
		String line = "";
		LinkedHashSet<String> lines = new LinkedHashSet<String>();
		try {
			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {
				if (!line.contains("NaN")) {
					lines.add(line);
				}
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
					BufferedWriter out = new BufferedWriter(new FileWriter(csvFile, false));
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
	}

	/*
	 * Sorts data, Removes duplicates
	 */
	public static void clean_data(String raw_data) {
		System.out.println("Cleaning...");
		for (File filename: new File(raw_data).listFiles()) { 
			if (filename.getName().endsWith(".csv")) {
				System.out.println(filename.getName());
				int activity = 4;
				clean_file(filename.getAbsolutePath());
				List<List<String>> csvLines = new ArrayList<List<String>>();
				CSVReader reader = null;
				String[] header = null;
				try {
					reader = new CSVReader(new FileReader(filename));
					header = reader.readNext();

					for (int key = 0; key < header.length; key++) {
						if (header[key].matches("Activity")) {
							activity = key;
						}
					}

					String[] line;
					while ((line = reader.readNext()) != null) {
						switch (line[activity]) {
						case "Nothing":
							line[activity] = "Inactive";
							break;
						case "Chin scratching":
							line[activity] = "Scratch Chin";
							break;
						case "Hair combing":
							line[activity] = "Comb Hair";
							break;
						case "Drawing":
							line[activity] = "Draw";
							break;
						case "Teeth brushing":
							line[activity] = "Brushing Teeth";
							break;
						case "Hand washing":
							line[activity] = "Washing Hands";
							break;
						case "Wash Hands":
							line[activity] = "Washing Hands";
						case "Medicating":
							line[activity] = "Take Medication";
							break;
						}
						csvLines.add(Arrays.asList(line));
					}
					reader.close();
					Comparator<List<String>> comp = new Comparator<List<String>>() {
						public int compare(List<String> csvLine1, List<String> csvLine2) {
							return Long.valueOf(csvLine1.get(0)).compareTo(Long.valueOf(csvLine2.get(0)));
						}
					};
					Collections.sort(csvLines, comp);
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					if (reader != null) {
						try {
							reader.close();
							CSVWriter writer = new CSVWriter(new FileWriter(filename));
							writer.writeNext(header);
							Iterator<List<String>> it = csvLines.iterator();
							while(it.hasNext()) {
								Iterator <String> it2 = it.next().iterator();
								String[] line;
								List<String> temp_line = new ArrayList<String>();
								while (it2.hasNext()) {
									temp_line.add(it2.next());
								}
								line = new String[temp_line.size()];
								temp_line.toArray(line);
								writer.writeNext(line);;
							}
							writer.close();	
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}


	public static void csv2arff(String csvfile, String arfffile) {

		// load CSV
		CSVLoader loader = new CSVLoader();
		try {
			loader.setSource(new File(csvfile));

			Instances data = loader.getDataSet();

			// save ARFF
			ArffSaver saver = new ArffSaver();
			saver.setInstances(data);
			saver.setFile(new File(arfffile));
			saver.setDestination(new File(arfffile));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static void replace_gesture(File cur_file, String[] gesture_list) {
		CSVReader reader = null;
		String[] line;
		LinkedHashSet<String[]> lines = new LinkedHashSet<String[]>();
		try {
			reader = new CSVReader(new FileReader(cur_file));
			int activity = 4;
			String[] header = reader.readNext();
			lines.add(header); //header
			for (int key = 0; key < header.length; key++) {
				if (header[key].matches("Gesture")) {
					activity = key;
				}
			}
			while ((line = reader.readNext()) != null) {
				//if gesture on this line is not in the array defined above, replace it with 'Inactive'
				if (line[activity].matches("Wash Hands")) {
					line[activity] = "Washing Hands";
				}
				else if (!(Arrays.asList(gesture_list).contains(line[activity]))) {
					//System.out.println(line[activity]);
					line[activity] = "Inactive";
				}

				lines.add(line); 
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
					CSVWriter writer = new CSVWriter(new FileWriter(cur_file));
					Iterator<String[]> it = lines.iterator();
					while(it.hasNext()) {
						writer.writeNext(it.next());
					}
					writer.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

	}


	public static HashMap<String, Boolean> alphabetize(File cur_file) {

		HashMap<String, Boolean> contain_gestures = new HashMap<String, Boolean>();

		contain_gestures.put("Brushing Teeth", Boolean.FALSE);
		contain_gestures.put("Comb Hair", Boolean.FALSE);
		contain_gestures.put("Draw", Boolean.FALSE);
		contain_gestures.put("Inactive", Boolean.FALSE);
		contain_gestures.put("Scratch Chin", Boolean.FALSE);
		contain_gestures.put("Smoking", Boolean.FALSE);
		contain_gestures.put("Take Medication", Boolean.FALSE);
		contain_gestures.put("Wash Hands", Boolean.FALSE);

		int activity = 4;
		CSVReader reader = null;
		String[] line;
		String[] header = null;
		ArrayList<String[]> inactive_lines = new ArrayList<String[]>();
		ArrayList<String[]> brushteeth_lines = new ArrayList<String[]>();
		ArrayList<String[]> combhair_lines = new ArrayList<String[]>();
		ArrayList<String[]> takemeds_lines = new ArrayList<String[]>();
		ArrayList<String[]> washhands_lines = new ArrayList<String[]>();
		ArrayList<String[]> scratchchin_lines = new ArrayList<String[]>();
		ArrayList<String[]> draw_lines = new ArrayList<String[]>();
		ArrayList<String[]> smoking_lines = new ArrayList<String[]>();


		try {
			reader = new CSVReader(new FileReader(cur_file));
			header = reader.readNext();

			for (int key = 0; key < header.length; key++) {
				if (header[key].matches("Gesture")) {
					activity = key;
				}
			}
			while ((line = reader.readNext()) != null) {
				switch (line[activity]) {
				case "Inactive":
					inactive_lines.add(line);
					contain_gestures.put("Inactive", Boolean.TRUE);
					break;
				case "Brushing Teeth":
					brushteeth_lines.add(line);
					contain_gestures.put("Brushing Teeth", Boolean.TRUE);
					break;
				case "Comb Hair":
					combhair_lines.add(line);
					contain_gestures.put("Comb Hair", Boolean.TRUE);
					break;
				case "Take Medication":
					takemeds_lines.add(line);
					contain_gestures.put("Take Medication", Boolean.TRUE);
					break;
				case "Wash Hands":
					washhands_lines.add(line);
					contain_gestures.put("Wash Hands", Boolean.TRUE);
					break;
				case "Scratch Chin":
					scratchchin_lines.add(line);
					contain_gestures.put("Scratch Chin", Boolean.TRUE);
					break;
				case "Draw":
					draw_lines.add(line);
					contain_gestures.put("Draw", Boolean.TRUE);
					break;
				case "Smoking":
					smoking_lines.add(line);
					contain_gestures.put("Smoking", Boolean.TRUE);
					break;
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

					/*LinkedHashSet<String[]> lines = new LinkedHashSet<String[]>();
					lines.add(header);
					lines.addAll(brushteeth_lines);
					lines.addAll(combhair_lines);
					lines.addAll(draw_lines);
					lines.addAll(inactive_lines);
					lines.addAll(scratchchin_lines);
					lines.addAll(smoking_lines);
					lines.addAll(takemeds_lines);
					lines.addAll(washhands_lines);

					CSVWriter writer = new CSVWriter(new FileWriter(cur_file));
					Iterator<String[]> it = lines.iterator();
					while(it.hasNext()) {
						writer.writeNext(it.next());
					}
					writer.close();*/
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return contain_gestures;
	}


	public static void add_missing_data(HashMap<String, Boolean> train_gestures, HashMap<String, Boolean> test_gestures, File train_data, File test_data) {
		Iterator<Entry<String, Boolean>> train_it = train_gestures.entrySet().iterator();
		Iterator<Entry<String, Boolean>> test_it = test_gestures.entrySet().iterator();
		CSVReader train_reader = null;
		CSVReader test_reader = null;
		CSVWriter test_writer = null;
		String[] line;
		String[] header = null;
		int activity = 4;
		Boolean added = Boolean.FALSE;
		ArrayList<String[]> linestoadd = new ArrayList<String[]>();
		ArrayList<String[]> lines = new ArrayList<String[]>();
		while (train_it.hasNext() && test_it.hasNext()) {
			Entry<String, Boolean> train_pair = train_it.next();
			Entry<String, Boolean> test_pair = test_it.next();

			if (train_pair.getKey().toString().matches(test_pair.getKey().toString())) {

				if ((Boolean)train_pair.getValue()) {//if ((Boolean)test_pair.getValue() != (Boolean)train_pair.getValue()) {

					try {
						train_reader = new CSVReader(new FileReader(train_data));
						header = train_reader.readNext();
						for (int key = 0; key < header.length; key++) {
							if (header[key].matches("Gesture")) {
								activity = key;
							}
						}
						while ((line = train_reader.readNext()) != null && !added) {
							if (line[activity].matches(train_pair.getKey().toString())) {
								linestoadd.add(line);
								//test_writer = new CSVWriter(new FileWriter(test_data, true));
								//test_writer.writeNext(line);
								added = Boolean.TRUE;
								//test_writer.close();
							}
						}
						added = false;
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
			train_it.remove(); // avoids a ConcurrentModificationException
			test_it.remove(); // avoids a ConcurrentModificationException
		}
		try {
			test_reader = new CSVReader(new FileReader(test_data));
			header = test_reader.readNext();
			while ((line = test_reader.readNext()) != null) {
				lines.add(line);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			test_reader.close();
			test_writer = new CSVWriter(new FileWriter(test_data, false));
			test_writer.writeNext(header);
			Iterator<String[]> it_new = linestoadd.iterator();
			while (it_new.hasNext()) {

				String[] temp = it_new.next();
				test_writer.writeNext(temp);
			}
			Iterator<String[]> it = lines.iterator();
			while (it.hasNext()) {
				test_writer.writeNext(it.next());
			}
			test_writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		//alphabetize(train_data);
		//alphabetize(test_data);
	}

	public static void gesture_time(String gesture_name, String output_folder, String stats_folder) {
		final  int min_gesture_time = 45;
		final Double threshold = 0.75; 
		final int gesture_end_threshold = 15;

		BufferedReader br = null;
		String line = "";
		Double gesture_time = 0.0;
		Double predicted_gesture_time = 0.0;
		Double total_gesture_time = 0.0;
		Double inactive_time = 0.0;
		int line_num = 1;
		int gesturegroup_cnt = 0;
		int max_cnt = 0;
		Double time_interval = 1.0;
		Double start_time = 0.0;
		Boolean predicted_gesture = Boolean.FALSE;
		String accuracy = null;
		String gesture_accuracy = null;
		Double ges_inactive_count;
		Double ges_gesture_count;
		ArrayList<Integer> gesture_starts = new ArrayList<Integer>();

		//False Positive Variables
		int falsepositives = 0;
		int num_falsepos = 0;
		Boolean gesture_started = false;
		int inactivegroup_cnt = 0;
		Double total_inactive_time = 0.0;
		Double tot_gesture_time = 0.0;
		int max_inactive_cnt = 0;
		Boolean predicted_inactive = Boolean.FALSE;
		Double in_inactive_count;
		Double in_gesture_count;

		ArrayList<Integer> inactive_starts = new ArrayList<Integer>();
		for(File output : new File(folder + "/" + output_folder).listFiles()) {
			line_num = 0;
			total_gesture_time = 0.0;
			gesture_time = 0.0;
			predicted_gesture_time = 0.0;
			falsepositives = 0;
			num_falsepos = 0;
			inactive_starts.clear();
			gesture_starts.clear();
			in_inactive_count = 0.0;
			in_gesture_count = 0.0;
			ges_inactive_count = 0.0;
			ges_gesture_count = 0.0;
			gesture_started = false;
			predicted_inactive = false;
			total_inactive_time = 0.0;
			if (output.getName().endsWith(".csv")) {//("_" + gesture_name + "1.csv") || output.getName().endsWith("_" + gesture_name + "2.csv") || output.getName().endsWith("_" + gesture_name + ".csv")) { 
				System.out.println("File: " + output);
				for(int a = 0; a < master_gesture_list.length;a++) {
					String[] gesture_list = master_gesture_list[a];
					for (int i = 0; i < gesture_list.length; i++) {
						Classification.accuracy(output.getAbsolutePath(),gesture_list[i]);
						try {
							br = new BufferedReader(new FileReader(output));
							line = br.readLine();
							accuracy = line.split(",")[1];
							gesture_accuracy = line.split(",")[3];
							br.readLine();
							start_time = 0.0;
							//end_time = 0.0;
							while ((line = br.readLine()) != null) {
								line_num++;
								/* result[0] - actual
						   		result[1] - predicted */
								String predicted = line.split(",")[1].trim();
								String actual = line.split(",")[0].trim();

								if (actual.equalsIgnoreCase(gesture_list[i])) { //actual equals gesture
									total_gesture_time += time_interval;

									if (predicted.equalsIgnoreCase(gesture_list[i])) {
										gesturegroup_cnt++;
										ges_gesture_count++;
										if (gesture_time == 0) {
											if (!gesture_started) {
												start_time = (double) line_num;
												gesture_started = true;
											}
											gesture_time += time_interval;
											predicted_gesture_time += time_interval;
										} else {
											gesture_time += time_interval + inactive_time;
											predicted_gesture_time += time_interval + inactive_time;
											ges_inactive_count += inactive_time;
										}
										inactive_time = 0.0;
										if(gesture_time >= min_gesture_time) {
											predicted_gesture = Boolean.TRUE;

											if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {
												gesture_starts.add(start_time.intValue() + 2);
												gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
											}
											//gesture_starts.add(start_time.intValue() + 2);
											//gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
										}

									} else { //predicted is inactive

										inactive_time += time_interval;
										max_cnt = Math.max(max_cnt, gesturegroup_cnt);
										gesturegroup_cnt = 0;
										//System.out.println(gesture_time);

										if (inactive_time >= gesture_end_threshold) {
											if (predicted_gesture == Boolean.TRUE ) {

												//System.out.println(start_time.intValue() + 2);
												//System.out.println("\tAccuracy: " + (ges_gesture_count/(ges_gesture_count+ges_inactive_count)));

												if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {
													gesture_starts.add(start_time.intValue() + 2);
													gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
												}
												if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {//(max_cnt >= 30) {
													gesture_starts.add(start_time.intValue() + 2);
													gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
												}
												predicted_gesture = Boolean.FALSE;
												max_cnt = 0;
												ges_gesture_count = 0.0;
												ges_inactive_count = 0.0;
											}
											gesture_time = 0.0;
											gesture_started = false;
										}

										inactivegroup_cnt = 0;

										if(total_inactive_time >= min_gesture_time) {
											if (in_gesture_count/(in_gesture_count+in_inactive_count) > threshold) {//(max_inactive_cnt >= 30) {

												if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {
													gesture_starts.add(start_time.intValue() + 2);
													gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
												}
												predicted_inactive = Boolean.FALSE;
												inactive_starts.add((int) (line_num-total_inactive_time+2));
												inactive_starts.add(line_num+1);
												num_falsepos++;						
											}
										}
										total_inactive_time = 0.0;
										max_inactive_cnt = 0;
									}
								} else { //actual equals Inactive
									if (!predicted.equalsIgnoreCase(gesture_list[i])) {
										inactive_time += time_interval; //inactive time, this is analogous to gesture time
										max_cnt = Math.max(max_cnt, gesturegroup_cnt);
										gesturegroup_cnt = 0;

										//This if statement checks if gesture_end_threshold number of inactives have occurred, and then if 
										//enough gestures have occurred add to gesture start
										if (inactive_time >= gesture_end_threshold) { //if inactive time is greater than the time the person can be inactive
											if (predicted_gesture == Boolean.TRUE ) { //if predicted gesture is running

												if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {
													gesture_starts.add(start_time.intValue() + 2);
													gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
												}
												predicted_gesture = Boolean.FALSE;
												max_cnt = 0;
												ges_gesture_count = 0.0;
												ges_inactive_count = 0.0;
											}
											gesture_time = 0.0;
											gesture_started = false;
										}

										inactivegroup_cnt = 0;

										/*if(total_inactive_time >= 45) {
											if (in_gesture_count/(in_gesture_count+in_inactive_count) > threshold) {//(max_inactive_cnt >= 30) {
												predicted_inactive = Boolean.FALSE;
												inactive_starts.add((int) (line_num-total_inactive_time+2));
												inactive_starts.add(line_num+1);
												num_falsepos++;						
											}
										}*/
										//total_inactive_time = 0.0;
										max_inactive_cnt = 0;
									}

									/********Look for false positives*******/
									if (predicted.equalsIgnoreCase(gesture_list[i])) {
										if (gesture_started) {
											gesturegroup_cnt++;
											ges_gesture_count++;
											gesture_time += time_interval + inactive_time;
											predicted_gesture_time += time_interval + inactive_time;
											ges_inactive_count += inactive_time;

											inactive_time = 0.0;

											if(gesture_time >= min_gesture_time) {
												predicted_inactive = Boolean.TRUE;
												inactive_starts.add((int) (line_num-total_inactive_time-12));
												inactive_starts.add(line_num-13);
												num_falsepos++;
											}
										} else {
											falsepositives++;
											in_gesture_count++;
											inactivegroup_cnt++;
											if (total_inactive_time == 0) {
												total_inactive_time += time_interval;
											} else {
												total_inactive_time += time_interval + tot_gesture_time;
												in_inactive_count += tot_gesture_time;
											}
											tot_gesture_time = 0.0;
											if(total_inactive_time >= min_gesture_time) {		
												//System.out.println("\tFalse Pos Accuracy 1: " + (in_gesture_count/(in_gesture_count+in_inactive_count)));
												//System.out.println("\t" + in_gesture_count);
												//System.out.println("\t" + in_inactive_count);

												if (in_gesture_count/(in_gesture_count+in_inactive_count) > threshold) {//(max_inactive_cnt >= 30) {
													predicted_inactive = Boolean.TRUE;
													inactive_starts.add((int) (line_num-total_inactive_time-12));
													inactive_starts.add(line_num-13);
													num_falsepos++;
												}
											}
										}
									} else { //predicted equals Inactive
										tot_gesture_time += time_interval;

										max_inactive_cnt = Math.max(max_inactive_cnt, inactivegroup_cnt);
										inactivegroup_cnt = 0;

										if (tot_gesture_time >= gesture_end_threshold) {

											if(total_inactive_time >= min_gesture_time) {
												//System.out.println("\tFalse Pos Accuracy 2: " + (in_gesture_count/(in_gesture_count+in_inactive_count)));

												if (in_gesture_count/(in_gesture_count+in_inactive_count) > threshold) {//(max_inactive_cnt >= 30) {
													predicted_inactive = Boolean.TRUE;
												}
											}
											if (predicted_inactive == Boolean.TRUE ) {
												predicted_inactive = Boolean.FALSE;
												inactive_starts.add((int) (line_num-total_inactive_time-12));
												inactive_starts.add(line_num-13);
												num_falsepos++;
											}
											total_inactive_time = 0.0;
											in_gesture_count = 0.0;
											in_inactive_count = 0.0;
											max_inactive_cnt = 0;
										}
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

									//System.out.println(total_inactive_time);
									if (predicted_inactive == Boolean.TRUE ) {
										inactive_starts.add((int) (line_num-total_inactive_time + 3));
										inactive_starts.add(line_num+2);
										num_falsepos++;
									}

									if (predicted_gesture == Boolean.TRUE ) {
										if (ges_gesture_count/(ges_gesture_count+ges_inactive_count) > threshold) {//(max_cnt >= 30) {
											gesture_starts.add(start_time.intValue() + 2);
											gesture_starts.add(start_time.intValue()+gesture_time.intValue() + 1);
										}								
									}

									if (total_gesture_time > 15 || !inactive_starts.isEmpty()) {
										String filename = folder + "/" + stats_folder + "/" + output.getName();

										BufferedWriter stats = new BufferedWriter(new FileWriter(filename, false)); //change back to true eventually
										//Double end_time = start_time + predicted_gesture_time;
										stats.write("Filename: ," + output.getName() + "\n");
										stats.write("Gesture: ," + gesture_list[i] + "\n");
										stats.write("Total Actual Gesture Time: ," + total_gesture_time + "\n");
										//stats.write("Gesture Start Time: ," + start_time + "\n");
										//stats.write("Gesture End Time: ," + end_time + "\n");
										Iterator<Integer> it_ges = gesture_starts.iterator();
										int ges_cur_start = 0;
										//Boolean written = false;
										int ges_start = 0;
										int ges_end = 0;
										while(it_ges.hasNext()) {
											ges_start = it_ges.next();
											if (ges_cur_start != 0 && ges_cur_start != ges_start) {
												if ((ges_end - ges_cur_start) >= min_gesture_time) {
													stats.write("Gesture Start Time: ," + ges_cur_start + "\n");
													stats.write("Gesture End Time: ," + ges_end + "\n");
													stats.write("Total Algorithm Predicted Gesture Time: ," + (ges_end-ges_cur_start) + "\n");
													System.out.println("\t" + "Gesture Start Time: " + ges_cur_start);
													System.out.println("\t" + "Gesture End Time: " + ges_end);
													//written = true;
												}	
											}
											ges_end = it_ges.next();
											//System.out.println("Cur: " + ges_cur_start);
											//System.out.println("Start: " + ges_start);
											//if (!(ges_cur_start == ges_start)) {
											//written = false;
											if (ges_cur_start == 0) {
												ges_cur_start = ges_start;
											}

											if ((ges_end - ges_start) >= min_gesture_time) {
												stats.write("Predicted Gesture Start Time: ," + ges_start + "\n");
												stats.write("Predicted Gesture End Time: ," + ges_end + "\n");
												//stats.write("Total Algorithm Predicted Gesture Time: ," + (ges_end-ges_start) + "\n");
												//System.out.println("\t" + "Gesture Start Time: " + ges_start);
												//System.out.println("\t" + "Gesture End Time: " + ges_end);
												//written = true;
											}
											ges_cur_start = ges_start;

											//}
										}
										/*if (!written) {
											if ((ges_end - ges_start) >= min_gesture_time) {
												stats.write("Gesture Start Time: ," + ges_start + "\n");
												stats.write("Gesture End Time: ," + ges_end + "\n");
												stats.write("Total Algorithm Predicted Gesture Time: ," + (ges_end-ges_start) + "\n");
												System.out.println("\t" + "Gesture Start Time: " + ges_start);
												System.out.println("\t" + "Gesture End Time: " + ges_end);
											}
										}*/
										//stats.write("Total Predicted Gesture Time: ," + predicted_gesture_time + "\n");
										stats.write(gesture_list[i] + " False Positives: ," + falsepositives + "\n");
										stats.write(gesture_list[i] + " False Positive Gestures: ," + num_falsepos + "\n");
										System.out.println("\t" + gesture_list[i] + " False Positive Gestures: " + num_falsepos);
										Iterator<Integer> it = inactive_starts.iterator();
										int faux_pas_cur_start = 0;
										//Boolean false_written = false;
										int faux_pas_start = 0;
										int faux_pas_end = 0;
										while (it.hasNext()) {
											faux_pas_start = it.next();
											if (faux_pas_cur_start != 0 && faux_pas_cur_start != faux_pas_start) {
												if (((faux_pas_end - faux_pas_cur_start) >= min_gesture_time)) {
													stats.write(gesture_list[i] + " False Positive Start Time: ," + faux_pas_cur_start + "\n");
													stats.write(gesture_list[i] + " False Positive End Time: ," + faux_pas_end + "\n");
													System.out.println("\t" + gesture_list[i] + " False Positive Start Time: " + faux_pas_cur_start);
													System.out.println("\t" + gesture_list[i] + " False Positive End Time: " + faux_pas_end);
													//written = true;
												}	
											}
											faux_pas_end = it.next()-1;
											//if (!(faux_pas_cur_start == faux_pas_start)) {
											//false_written = false;

											faux_pas_cur_start = faux_pas_start;
											if ((faux_pas_end - faux_pas_start) >= min_gesture_time) {
												//stats.write("False Positive Start Time: ," + faux_pas_start + "\n");
												//stats.write("False Positive End Time: ," + faux_pas_end + "\n");
												//System.out.println("\t" + "False Positive Start Time: " + faux_pas_start);
												//System.out.println("\t" + "False Positive End Time: " + faux_pas_end);
												//false_written = true;
											}
										}
										//}

										/*if (!false_written) {
											if ((faux_pas_end - faux_pas_start) >= min_gesture_time) {
												stats.write("False Positive Start Time: ," + faux_pas_start + "\n");
												stats.write("False Positive End Time: ," + faux_pas_end + "\n");
												System.out.println("\t" + "False Positive Start Time: " + faux_pas_start);
												System.out.println("\t" + "False Positive End Time: " + faux_pas_end);
												false_written = true;
											}
										}*/
										stats.write("Gesture Accuracy: ," + gesture_accuracy + "\n");
										stats.write("Accuracy: ," + accuracy + "\n");
										stats.close();
										System.out.println("\n");
									}
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
					}
				}
			}
		}
	}


	/**
	 * Applies Tier II recognition and looks for True Positives, False Positives, and False Negatives
	 * 
	 * @param output_folder Folder containing the output files
	 * @param stats_folder  Folder where analysis results are stored
	 */
	public static void tierII_Analysis(String output_folder, String stats_folder) {

		BufferedReader br = null;
		String line = "";
		String accuracy = "";
		String gesture_accuracy = "";
		String actual = "";
		String predicted = "";

		final boolean ADD = true;
		final boolean REPLACE = false;

		//iterate through files in Output folder
		for(File output : new File(folder + "/" + output_folder).listFiles()) {
			System.out.println("File: " + output);

			double activity_instance_time = 0.0;
			double activity_instance_inactive_time = 0.0;

			int start_time = 0;
			int total_time = 3; //3 lines of header and weka

			Map<Integer, Integer> test_metrics = new HashMap<>();
			test_metrics.put(TRUE_POSITIVES, 0);
			test_metrics.put(FALSE_POSITIVES, 0);
			test_metrics.put(TRUE_NEGATIVES, 0);
			test_metrics.put(FALSE_NEGATIVES, 0);

			ArrayList<activity_instance> activity_instances = new ArrayList<>(); //TP, FP, TN, FN; start window; end window		
			ArrayList<Integer> cur_activity_sequence = new ArrayList<>(); //sequence of tier I windows as 0 or 1

			for(int a = 0; a < master_gesture_list.length;a++) {
				String[] gesture_list = master_gesture_list[a];
				for (int i = 0; i < gesture_list.length; i++) {
					Classification.accuracy(output.getAbsolutePath(),gesture_list[i]);
					try {
						br = new BufferedReader(new FileReader(output));
						line = br.readLine();
						accuracy = line.split(",")[1];
						gesture_accuracy = line.split(",")[3];
						br.readLine(); //Actual Class, Predicted
						br.readLine(); //added for weka

						//Iterate through file
						while ((line = br.readLine()) != null) {

							total_time++;
							actual = line.split(",")[0].trim();
							predicted = line.split(",")[1].trim();

							if (actual.matches(predicted)) {
								if (actual.matches("Inactive")) { //Tier I True Negative

									test_metrics.put(TRUE_NEGATIVES, test_metrics.get(TRUE_NEGATIVES)+1);
									if (activity_instance_time != 0.0) {
										activity_instance_inactive_time++;
										cur_activity_sequence.add(INACTIVE);
									}

								} else { //Tier I True Positive
									test_metrics.put(TRUE_POSITIVES, test_metrics.get(TRUE_POSITIVES)+1);
									if (activity_instance_time == 0.0) {
										start_time = total_time;
									}

									cur_activity_sequence.add(ACTIVITY);
									activity_instance_time += time_interval + activity_instance_inactive_time;
									activity_instance_inactive_time = 0;
								}
							} else { //actual != predicted
								if (actual.matches("Inactive")) { //Tier I False Positive

									test_metrics.put(FALSE_POSITIVES, test_metrics.get(FALSE_POSITIVES)+1);
									if (activity_instance_time == 0.0) {
										start_time = total_time;
									}

									cur_activity_sequence.add(ACTIVITY);
									activity_instance_time += time_interval + activity_instance_inactive_time;
									activity_instance_inactive_time = 0;

								} else { //Tier I False Negative

									test_metrics.put(FALSE_NEGATIVES, test_metrics.get(FALSE_NEGATIVES)+1);

									if (activity_instance_time != 0.0) {
										activity_instance_inactive_time++;
										cur_activity_sequence.add(INACTIVE);
									}
								}
							}

							//Check if the activity happened
							if (activity_instance_time != 0) {
								int num_activity = Collections.frequency(cur_activity_sequence, ACTIVITY);
								int end_time = total_time;
								if (activity_instance_inactive_time >= gesture_end_threshold || num_activity/(cur_activity_sequence.size()) > activity_percentage_threshold) {
									if (activity_instance_time >= min_gesture_time) {
										while(cur_activity_sequence.get(cur_activity_sequence.size()-1) == INACTIVE) {
											end_time--;
											cur_activity_sequence.remove(cur_activity_sequence.size()-1);
										}

										double range = time_interval*(cur_activity_sequence.size()-1);
										if (range >= min_gesture_time) {

											if (activity_instances.isEmpty()) {
												add_instance(ADD, activity_instances, start_time, end_time, test_metrics);
											} else {
												if (activity_instances.get(activity_instances.size()-1).start_time == start_time) {
													if (activity_instances.get(activity_instances.size()-1).end_time < end_time) {
														add_instance(REPLACE, activity_instances, start_time, end_time, test_metrics);
													}
												} else {
													add_instance(ADD, activity_instances, start_time, end_time, test_metrics);
												}
											}
										}
									}


									if (activity_instance_inactive_time >= gesture_end_threshold) {
										cur_activity_sequence.clear();
										activity_instance_time = 0.0;
										activity_instance_inactive_time = 0;
									}
								}
							}

						}

						if (!cur_activity_sequence.isEmpty()) { //check at the end of the file for gesture since you'll never reach inactive time threshold
							if (activity_instance_time >= min_gesture_time) {
								while(cur_activity_sequence.get(cur_activity_sequence.size()-1) == INACTIVE) {
									cur_activity_sequence.remove(cur_activity_sequence.size()-1);
								}
								double range = time_interval*(cur_activity_sequence.size()-1);
								if (range >= min_gesture_time) {
									add_instance(ADD, activity_instances, start_time, total_time, test_metrics);
								}
							}
							cur_activity_sequence.clear();
						}

					} catch (FileNotFoundException e) {
						e.printStackTrace();
					} catch (IOException e) {
						e.printStackTrace();
					} finally {
						if (br !=null) {
							try {
								br.close();

								//Write to File

								String filename = folder + "/" + stats_folder + "/" + output.getName();


								BufferedWriter stats = new BufferedWriter(new FileWriter(filename, false));
								stats.write("Filename:," + output.getName() + "\n");
								stats.write("Gesture:," + gesture_list[i] + "\n");
								stats.write("Tier I Recognition Accuracy:," + accuracy + "\n");
								stats.write("Tier I Gesture Accuracy:," + gesture_accuracy + "\n");
								stats.newLine();

								for (activity_instance instance : activity_instances) {
									stats.newLine();
									stats.write("Result:," + instance.type + "\n");
									stats.write("Start Time:," + instance.start_time + "\n");
									stats.write("End Time:," + instance.end_time + "\n");
								}

								stats.close();

							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					}
				}
			}
		}
	}

	/**
	 * Adds the activity to the ArrayList of activity instances
	 * 
	 * @param add_replace add is true, replace is false
	 * @param activity_instances ArrayList of activity instances
	 * @param start_time Start time of activity
	 * @param end_time End time of activity
	 * @param cur_activity_sequence Sequence of tierI results
	 * @param test_metrics Sequence of TP, FP, TN, FN for activity 
	 */
	public static void add_instance(boolean add_replace, ArrayList<activity_instance> activity_instances, int start_time, int end_time, Map<Integer, Integer> test_metrics) {

		int true_positives = test_metrics.get(TRUE_POSITIVES);
		int false_positives = test_metrics.get(FALSE_POSITIVES);
		int true_negatives = test_metrics.get(TRUE_NEGATIVES);
		int false_negatives = test_metrics.get(FALSE_NEGATIVES);
		int num_results = true_positives + false_positives + true_negatives + false_negatives;

		if ((false_negatives + true_positives + false_positives) >= min_gesture_time && (false_negatives + true_positives + false_positives)/num_results > activity_percentage_threshold) {

			if (true_positives > min_gesture_time) {
				if (add_replace) {
					activity_instances.add(new activity_instance("True Positive", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("True Positive", start_time, end_time));
				}
			} else if (false_negatives + true_positives > min_gesture_time) {
				if (add_replace) {
					activity_instances.add(new activity_instance("False Negative", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("False Negative", start_time, end_time));
				}
			} else if (false_positives > min_gesture_time) {
				if (add_replace) {
					activity_instances.add(new activity_instance("False Positive", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("False Positive", start_time, end_time));
				}
			}
		}





		/*if (true_positives < min_gesture_time && (true_positives+false_positives) >= min_gesture_time && true_positives > false_positives) {
			if (add_replace) {
				//System.out.println(x);
				activity_instances.add(new activity_instance("False Negative222", start_time, end_time));
			} else {
				activity_instances.set(activity_instances.size()-1, new activity_instance("False Negative", start_time, end_time));
			}
		} else if (false_positives > min_gesture_time || ((true_positives + false_positives) >= min_gesture_time && true_positives < min_gesture_time)) {
			if (add_replace) {
				activity_instances.add(new activity_instance("False Positive", start_time, end_time));
			} else {
				activity_instances.set(activity_instances.size()-1, new activity_instance("False Positive", start_time, end_time));
			}
		} else if ((true_positives + false_negatives + false_positives) > min_gesture_time) {
			if (true_positives/(true_positives+false_negatives + false_positives + true_negatives) > activity_percentage_threshold) {
				if (add_replace) {
					activity_instances.add(new activity_instance("True Positive", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("True Positive", start_time, end_time));
				}
			} else {
				if (add_replace) {
					activity_instances.add(new activity_instance("False Negative", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("False Negative", start_time, end_time));
				}
			}
		}*/


		/*int num_activity = Collections.frequency(cur_activity_sequence, ACTIVITY);

		if (num_activity/(cur_activity_sequence.size()-1) > activity_percentage_threshold) {

			Map.Entry<Integer, Integer> maxEntry = null;
			for (Map.Entry<Integer, Integer> entry : test_metrics.entrySet()) {
				if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0) {
					maxEntry = entry;
				}
			}

			switch(maxEntry.getKey()) {
			case TRUE_POSITIVES:
				if (add_replace) {
					activity_instances.add(new activity_instance("True Positive", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("True Positive", start_time, end_time));
				}
				break;
			case FALSE_POSITIVES:
				if (add_replace) {
					activity_instances.add(new activity_instance("False Positive", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("False Positive", start_time, end_time));
				}
				break;
			case TRUE_NEGATIVES:
				if (add_replace) {
					activity_instances.add(new activity_instance("True Negative", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("True Negative", start_time, end_time));
				}
				break;
			case FALSE_NEGATIVES:
				if (add_replace) {
					activity_instances.add(new activity_instance("False Negative", start_time, end_time));
				} else {
					activity_instances.set(activity_instances.size()-1, new activity_instance("True Negative", start_time, end_time));
				}
				break;
			}
		}*/
	}

	public static void remove_incorrect(ArrayList<Integer> incorrect, String train_data) {

		System.out.println("Incorrect: " + incorrect.size());
		CSVReader reader = null;
		String[] line = null;
		ArrayList<String[]> csvLines = new ArrayList<String[]>();
		try {
			reader = new CSVReader(new FileReader(train_data));
			int train_index = 1;
			csvLines.add(reader.readNext()); //header
			while ((line = reader.readNext()) != null) {
				if (!incorrect.contains(train_index)) {
					csvLines.add(line);
				}
				train_index++;
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
					CSVWriter out = new CSVWriter(new FileWriter(train_data, false));
					Iterator<String[]> it = csvLines.iterator();
					while(it.hasNext()) {
						out.writeNext(it.next());
					}
					out.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
}

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class misc {

	final static String[][] master_gesture_list = {{"Washing Hands"}};//,"Comb Hair","Drinking","Scratch Chin","Take Medication","Wash Hands"}};
	final static String raw_data = "raw data";
	final static String folder = "data/thesis";
	final static String test_folder = "Test Data/Brushing Teeth";
	final static String output_folder = "Outputs/Brushing Teeth";
	final static String stats_folder = "Stats/Brushing Teeth/stats_brushteeth_original_RF";
	final static String false_positives_folder = "False Positives Original RF";
	final static String graph_data = "data/graph_data";

	final static int TRUE_POSITIVES = 0;
	final static int FALSE_POSITIVES = 1;
	final static int TRUE_NEGATIVES = 2;
	final static int FALSE_NEGATIVES = 3;

	final static int ACTIVITY = 0;
	final static int INACTIVE = 1;

	//Tier II Parameters
	final  static int min_gesture_time = 45;
	final static double activity_percentage_threshold = 0.75;
	final static int gesture_end_threshold = 15;
	final static double time_interval = 1.0;

	public static void calculate_average_time() {
		Double average_time = 0.0;
		ArrayList<Double> x = new ArrayList<Double>();
		Double std = 0.0;
		for (File test_data_file : new File(raw_data).listFiles()) {
			Double start_time = 0.0;
			Double end_time = 0.0;
			if (test_data_file.getName().endsWith(".csv")) {
				CSVReader reader = null;
				String[] line = null;
				try {
					reader = new CSVReader (new FileReader(test_data_file));
					line = reader.readNext();
					line = reader.readNext();
					start_time = Double.parseDouble(line[0])/1000;
					while ((line = reader.readNext()) != null) {
						end_time = Double.parseDouble(line[0])/1000;
					}

				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {

					if ((end_time - start_time) > 30) {
						System.out.println("Calculating: " + test_data_file.getName());

						//System.out.println("Start Time: " + start_time);
						//System.out.println("End Time: " + end_time);
						//System.out.println("Minutes: " + (end_time - start_time)/60);
						//System.out.println("Hours: " + (end_time - start_time)/3600);
						x.add(end_time-start_time);
						average_time = average_time + (end_time-start_time);
					}
					if (reader != null) {
						try {
							reader.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
		System.out.println("Total Time: " + average_time/60/60);
		System.out.println("Average Time: " + average_time/9/6.25/60/60);
		Double avg = average_time/9/6.25/60/60;
		for (int i = 0; i < x.size(); i++) {
			std +=Math.pow((x.get(i) - avg),2);
		}
		System.out.println("Standard Deviation: " + Math.sqrt(std/x.size())/3600/6.25);
	}

	public static void graph_false_positives(File stats_folder) {

		CSVReader stats_reader = null;
		CSVReader test_reader = null;
		CSVReader raw_reader = null;
		CSVWriter false_pos_writer = null;
		String[] line;
		String[] test_line;
		String[] raw_line;
		String[] test_header;
		String[] raw_header;
		int start_time_col = 0;
		int end_time_col = 0;
		int time_col = 0;
		int start = 0;
		int end = 0;
		int filename_num = 1;
		Double start_time = 0.0;
		Double end_time = 0.0;
		LinkedHashSet<String[]> raw_lines = new LinkedHashSet<String[]>();
		String false_pos_filename;
		for (File stats_file : stats_folder.listFiles()) {
			if (stats_file.getName().endsWith(".csv")) {
				System.out.println(stats_file.getName());
				try {
					stats_reader = new CSVReader(new FileReader(stats_file));
					while (!(line = stats_reader.readNext())[0].contains("False Positive Gestures")) {
					}
					int false_pos = Integer.parseInt(line[1]); //False Positive Gestures
					if (false_pos > 0) {
						false_pos_filename = stats_file.getName();
						test_reader = new CSVReader(new FileReader(folder + "/" + test_folder + "/" + stats_file.getName()));

						test_header = test_reader.readNext();
						for (int key = 0; key < test_header.length; key++) {
							if (test_header[key].matches("Start Time")) {
								start_time_col = key;
							} else if (test_header[key].matches("End Time")) {
								end_time_col = key;
							}
						}

						while (!(line = stats_reader.readNext())[0].contains("Gesture Accuracy:")) {
							if (line[0].contains("Start Time")) {
								start = Integer.parseInt(line[1])-2;
								end = Integer.parseInt(stats_reader.readNext()[1])-2;
								int cur_line = 1;
								while (cur_line <= end && (test_line = test_reader.readNext()) != null) {
									if (cur_line == start) {
										start_time = Double.parseDouble(test_line[start_time_col])*1000;
										System.out.println("Start Time: " + start_time);
									} else if (cur_line == end) {
										end_time = Double.parseDouble(test_line[end_time_col])*1000;
										System.out.println("End Time: " + end_time);
									}
									cur_line++;
								}

								//look for start_time and end_time in raw data
								raw_reader = new CSVReader(new FileReader(raw_data + "/" + stats_file.getName().substring(master_gesture_list[0][0].length()+2)));
								raw_header = raw_reader.readNext();
								for (int key = 0; key < raw_header.length; key++) {
									if (raw_header[key].matches("PebbleAccT")) {
										time_col = key;
									}
								}

								Boolean extract = false;
								raw_lines.clear();
								while ((raw_line = raw_reader.readNext()) != null) {
									if (Double.parseDouble(raw_line[time_col]) == start_time) {
										extract = true;
									} else if(Double.parseDouble(raw_line[time_col]) == end_time) {
										extract = false;
									}
									if (extract) {
										if (filename_num == 1) {
											//System.out.println(raw_line[0]);
										}
										raw_lines.add(raw_line);
									}
								}

								false_pos_writer = new CSVWriter(new FileWriter(folder + "/" + false_positives_folder + "/" + false_pos_filename));
								false_pos_filename = stats_file.getName().substring(0, stats_file.getName().length()-4) + filename_num + ".csv";
								filename_num++;
								Iterator<String[]> raw_it = raw_lines.iterator();
								false_pos_writer.writeNext(raw_header);
								while (raw_it.hasNext()) {
									false_pos_writer.writeNext(raw_it.next());
								}
								raw_lines.clear();
								false_pos_writer.close();
							}
						}

						if (raw_reader != null) {
							try {
								raw_reader.close();
							} catch (IOException e) {
								e.printStackTrace();
							}						}

						if (test_reader != null) {
							try {
								test_reader.close();
							} catch (IOException e) {
								e.printStackTrace();
							}						}
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					if (stats_reader != null) {
						try {
							stats_reader.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}

	public static void extract_gesture_data() {

		CSVReader raw_reader = null;
		CSVWriter gesture_writer = null;
		String[] raw_line;
		String[] raw_header;
		Boolean started;
		String cur_filename;
		int file_num;
		Boolean gesture_writer_open = false;

		for (File raw_file : new File(raw_data).listFiles()) {
			if (raw_file.getName().endsWith(".csv")) {
				started = false;
				file_num = 1;
				try {
					raw_reader = new CSVReader(new FileReader(raw_file));
					raw_header = raw_reader.readNext();
					while ((raw_line = raw_reader.readNext()) != null) {
						if (!raw_line[raw_line.length-1].contains("Inactive")) {
							if (!started) {
								cur_filename = raw_file.getName().substring(0, raw_file.getName().length()-4) + raw_line[raw_line.length-1].replaceAll("\\s+","") + ".csv";
								while (new File(cur_filename).exists()) {
									cur_filename = raw_file.getName().substring(0, raw_file.getName().length()-4) + raw_line[raw_line.length-1].replaceAll("\\s+","") + "_" + file_num + ".csv";
									file_num++;
								}
								if (gesture_writer_open) {
									gesture_writer.close();
								}
								gesture_writer = new CSVWriter(new FileWriter("Gestures Data/" + cur_filename));
								gesture_writer_open = true;
								gesture_writer.writeNext(raw_header);
								started = true;
							}
							gesture_writer.writeNext(raw_line);
						} else {
							started = false;
						}
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					if (raw_reader != null) {
						try {
							raw_reader.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}

	public static double count_gestures(File cur_file) {

		CSVReader reader = null;
		String[] header = null;
		String[] line = null;
		int activity = 4;

		int gesture_lines = 0;

		try {
			reader = new CSVReader(new FileReader(cur_file));
			header = reader.readNext();

			for (int key = 0; key < header.length; key++) {
				if (header[key].matches("Activity")) {
					activity = key;
				}
			}
			while ((line = reader.readNext()) != null) {

				if (line[activity].contains("Wash Hands")) {
					gesture_lines++;
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

		return gesture_lines*0.04;
	}

	public static void percent_check(File stats_folder) {
		CSVReader stats_reader = null;
		CSVReader test_reader = null;
		CSVReader output_reader = null;
		CSVWriter false_pos_writer = null;
		String[] line;
		String[] test_line;
		String[] raw_line;
		String[] test_header;
		String[] raw_header;
		int start_time_col = 0;
		int end_time_col = 0;
		int time_col = 0;
		int start = 0;
		int end = 0;
		int filename_num = 1;
		Double start_time = 0.0;
		Double end_time = 0.0;
		LinkedHashSet<String[]> raw_lines = new LinkedHashSet<String[]>();
		String false_pos_filename;
		for (File stats_file : stats_folder.listFiles()) {
			if (stats_file.getName().endsWith(".csv")) {
				System.out.println(stats_file.getName());
				try {
					stats_reader = new CSVReader(new FileReader(stats_file));
					line = stats_reader.readNext();

					while (!(line[0].contains("False Positive Gestures") || line[0].contains("Gesture Start Time"))) {
						line = stats_reader.readNext();
					}

					if (line[0].contains("False Positive Gestures")) {
						if (!line[1].matches("0")) {
							line = stats_reader.readNext();
							start_time = Double.parseDouble(line[1]);
							line = stats_reader.readNext();
							end_time = Double.parseDouble(line[1]);


						}
					} else if (line[0].contains("Gesture Start Time")) {

					}

					while ((line = stats_reader.readNext()) != null) {




					}

				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					if (stats_reader != null) {
						try {
							stats_reader.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}

	public static void first_stage_total_time() {
		CSVReader reader = null;
		String[] line;
		long start = 0;
		long end = 0;
		long total = 0;
		for (File raw_file : new File("Train Data").listFiles()) {
			if (raw_file.getName().contains("_") && !raw_file.getName().contains("_draw")) {
				System.out.println(raw_file.getName());
				try {
					reader = new CSVReader(new FileReader(raw_file));
					line = reader.readNext();
					start = Long.parseLong(line[0]);
					end = start;
					while ((line = reader.readNext()) != null) {
						end = Long.parseLong(line[0]);
					}
					total += (end-start);

				} catch (NumberFormatException | IOException e) {
					e.printStackTrace();
				}
			}

		}

		System.out.println("Total time: " + total);
	}

	public static void combine_csv(File training_data, File add_data) {
		CSVReader reader = null;
		CSVWriter writer = null;
		String[] line = null;

		try {
			reader = new CSVReader(new FileReader(add_data));
			writer = new CSVWriter(new FileWriter(training_data, true));
			reader.readNext(); //header

			while ((line = reader.readNext()) != null) {
				writer.writeNext(line);
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
			if (writer != null) {
				try {
					writer.close();

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static void phase_three_performance() {
		//find average for each folder
		CSVReader reader = null;
		String[] line;
		String actual;
		String predicted;
		for (File output_subfolder : new File(folder + "/" + output_folder).listFiles()) {
			if (output_subfolder.isDirectory() && output_subfolder.getName().contains("new")) {
				System.out.println(output_subfolder.getName());
				double tp_cnt = 0;
				double tn_cnt = 0;
				double fp_cnt = 0;
				double fn_cnt = 0;

				for (File output_file : output_subfolder.listFiles()) {
					try {
						reader = new CSVReader(new FileReader(output_file));
						line = reader.readNext(); //Accuracy
						line = reader.readNext(); //header
						line = reader.readNext(); //Brush teeth training data added line
						line = reader.readNext(); //Inactive training data added line

						while ((line = reader.readNext()) != null) {
							actual = line[0];
							predicted = line[1];

							if (actual.contains("Brushing Teeth")){
								if (predicted.contains("Brushing Teeth")) {
									tp_cnt++;
								} else if (predicted.contains("Inactive")) {
									fn_cnt++;
								}
							} else if (actual.contains("Inactive")) {
								if (predicted.contains("Brushing Teeth")) {
									fp_cnt++;
								} else if (predicted.contains("Inactive")) {
									tn_cnt++;
								}
							}
						}


					} catch (FileNotFoundException e) {
						e.printStackTrace();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				System.out.println("Accuracy: " + (tp_cnt + tn_cnt)/(tp_cnt + tn_cnt + fp_cnt + fn_cnt)*100);

				System.out.println("F-measure: " + (2*tp_cnt)/(2*tp_cnt + fp_cnt + fn_cnt));
			}
		}

	}

	//checks how much is corrupted
	public static void raw_data_corrupted() {
		Double num_total_lines = 0.0;
		Double num_corrupted_lines = 0.0;
		Double prev_time;
		Double cur_time;
		Double diff;
		CSVReader reader = null;
		String[] header = null;
		String[] line = null;
		int time_col = 0;

		for (File raw_data_file : new File(raw_data).listFiles()) {
			if (raw_data_file.getName().endsWith(".csv")) {
				//check difference between
				//System.out.println(raw_data_file.getName());
				prev_time = 0.0;
				cur_time = 0.0;
				try {
					reader = new CSVReader(new FileReader(raw_data_file));
					header = reader.readNext();
					for (int key = 0; key < header.length; key++) {
						if (header[key].matches("PebbleAccT")) {
							time_col = key;
						}
					}
					while ((line = reader.readNext()) != null) {
						num_total_lines++;
						if (prev_time == 0) {
							prev_time = Double.parseDouble(line[time_col]);
						} else {
							cur_time = Double.parseDouble(line[time_col]);
							diff = cur_time - prev_time;
							if (diff < 35 || diff > 45) {
								//System.out.println(raw_data_file.getName() + ": " + diff);
								num_corrupted_lines++;
							}
							prev_time = cur_time;
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
		}
		System.out.println("Total Data Entries: " + num_total_lines);
		System.out.println("Total Corrupted Entries: " + num_corrupted_lines);

		Double percentage = (num_corrupted_lines/num_total_lines)*100;
		System.out.println("Percentage Corrupted: " + percentage + "%");
	}

	public static void separate_graphs(String graph_folder) {

		CSVReader reader = null;
		CSVWriter writer = null;
		String[] line;
		String cur_gesture;
		int file_lines = 0;
		for (File filename: new File(graph_folder).listFiles()) {
			if (filename.getName().endsWith(".csv")) {
				System.out.println(filename.getName());
				int activity = 4;
				String[] header = ("PebbleAccT,PebbleAccX,PebbleAccY,PebbleAccZ").split(",");
				int file_num = 0;
				try {
					reader = new CSVReader(new FileReader(filename));
					//header = reader.readNext();
					//for (int key = 0; key < header.length; key++) {
					//	if (header[key].matches("Activity")) {
					//		activity = key;
					//	}
					//}

					line = reader.readNext();
					while (line != null) {
						cur_gesture = line[activity];
						file_lines = 0;
						if (cur_gesture.contains("Brushing Teeth")) {
							String write_file = graph_data + "/" + filename.getName().substring(0, filename.getName().length()-4) + "_" + file_num + "_" + cur_gesture + ".csv";
							file_num++;
							writer = new CSVWriter(new FileWriter(write_file),',','\0');
							writer.writeNext(Arrays.copyOfRange(header, 0, 4));
	
							while (line != null && line[activity].matches(cur_gesture)) {							
								writer.writeNext(Arrays.copyOfRange(line, 0, 4));
								line = reader.readNext();
								file_lines++;
							}
							writer.close();
							if (file_lines < 10) {
								new File(write_file).delete();
								file_num--;
							}
						} else {
							while (line != null && line[activity].matches(cur_gesture)) {							
								line = reader.readNext();
							} 
						}
					}
				} catch (IOException e) {
					e.printStackTrace();
				} 
			}
		}
		System.out.println("Finished Extracting Gesture Data");
	}
}

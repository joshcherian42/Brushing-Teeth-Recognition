import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class extract_features {
	final static double window_size = 4;
	final static double overlap_size = 1;
	final static double sampling_rate = 25;

	public static Double[] generate_features (File folder, String output_path, Double[] window) {
		
		if (folder.getAbsolutePath().contains("Train")) {
			window = gen_features(folder, output_path, Boolean.TRUE, window[0], window[1], window[2]);

		} else if (folder.getAbsolutePath().contains("raw data")) {
			
			window = gen_features(folder, output_path, Boolean.FALSE, window[0], window[1], window[2]);
			
		}

		return window;
	}

	/*Find most common Gesture in window*/
	public static String win_gesture (ArrayList<String> gestures) {
		String gesture = "Inactive";
		HashMap<String, Integer> gestures_cnt = new HashMap<>();
		for (int c = 0; c < gestures.size(); c++) {
			if (gestures_cnt.containsKey(gestures.get(c))) {
				gestures_cnt.put(gestures.get(c),gestures_cnt.get(gestures.get(c))+1);
			} else {
				gestures_cnt.put(gestures.get(c), 1);
			}
		}
		for (String key : gestures_cnt.keySet()){
			int max_value = 0;
			int cur_value = gestures_cnt.get(key);
			if (cur_value > max_value) {
				max_value = cur_value;
				gesture = key;
			}
		}
		return gesture;
	}

	/**
	 * 
	 * @param fileEntry Raw Data file
	 * @param output_path File to write features too
	 * @param append Boolean whether or not to overwrite/append features file
	 * @param total_windows
	 * @param corrupted_windows
	 * @param corrupted_gesture_windows
	 * @return
	 */
	public static Double[] gen_features (File fileEntry, String output_path, Boolean append, Double total_windows, Double corrupted_windows, Double corrupted_gesture_windows) {

		/*Header*/
		String[] header = (
				"Avg Jerk X,Avg Jerk Y,Avg Jerk Z,"
						+ "Avg Height X,Avg Height Y,Avg Height Z,"
						+ "Stdev Height X,Stdev Height Y,Stdev Height Z,"
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
						+ "Gesture,"
						+ "Start Time,End Time").split(",");
		CSVWriter writer;
		CSVReader reader_output; //checks if header is already written to file
		try {
			writer = new CSVWriter(new FileWriter(output_path, append));

			if (append) {
				reader_output = new CSVReader(new FileReader(output_path));
				if ((reader_output.readNext()) == null) {
					writer.writeNext(header);
				}
				reader_output.close();
			} else {
				writer.writeNext(header);
			}
			String[] nextLine;
			int dyn_init = 0;

			//System.out.println(fileEntry);
			String cur_gesture = "Inactive";
			//String prev_gesture = "Inactive";
			if (fileEntry.getName().endsWith(".csv") && !fileEntry.getName().endsWith("draw.csv")) {
				CSVReader reader = new CSVReader(new FileReader(fileEntry), ',', '"', 1);
				ArrayList<Double> time = new ArrayList<Double>();
				int cur_length = 0;
				Double init_time = 0.0;
				Double last_time = 0.0;
				int shift_mult = 1;
				ArrayList<Double> x = new ArrayList<Double>();
				ArrayList<Double> y = new ArrayList<Double>();
				ArrayList<Double> z = new ArrayList<Double>();
				ArrayList<String> gestures = new ArrayList<String>();
				nextLine = reader.readNext();

				while (nextLine != null) {
					Double cur_time = Double.parseDouble(nextLine[0])/1000;
					Double prev_time;
					Double total_time = 0.0;

					//first window_size
					while (total_time < window_size && dyn_init == 0) {
						if (nextLine != null) {
							cur_time = Double.parseDouble(nextLine[0])/1000;

							time.add(Double.parseDouble(nextLine[0])/1000);

							//first time or not
							if (total_time == 0 && cur_length == 0) {
								prev_time = time.get(cur_length);

							} else if (total_time == 0 && last_time != 0.0){
								prev_time = last_time;
							}
							else {
								prev_time = time.get(cur_length - 1);
							}
							total_time = total_time + (cur_time - prev_time);

							x.add(Double.parseDouble(nextLine[1]));
							y.add(Double.parseDouble(nextLine[2]));
							z.add(Double.parseDouble(nextLine[3]));
							gestures.add(nextLine[4]);


							if (total_time >= window_size) {
								last_time = time.get(cur_length);
								nextLine = reader.readNext();
							}
							cur_length++;

						} else {
							break;
						}
						if (total_time < window_size) {
							nextLine = reader.readNext();
						}
					} //end initial second

					if (dyn_init == 1) {

						if (nextLine != null) {
							while (time.get(0) < init_time + (overlap_size*shift_mult)) {
								time.remove(0);
								x.remove(0);
								y.remove(0);
								z.remove(0);
								gestures.remove(0);
								if (x.size() == 0 ) {
									time.add(Double.parseDouble(nextLine[0])/1000);
									x.add(Double.parseDouble(nextLine[1]));
									y.add(Double.parseDouble(nextLine[2]));
									z.add(Double.parseDouble(nextLine[3]));
									gestures.add(nextLine[4]);
									nextLine = reader.readNext();
									while (init_time + (overlap_size*shift_mult) < cur_time) {
										shift_mult++;
									}
									shift_mult--;
									break;
								}
							}
							while (cur_time <= init_time + (overlap_size*shift_mult) + window_size && nextLine != null) {
								time.add(Double.parseDouble(nextLine[0])/1000);
								x.add(Double.parseDouble(nextLine[1]));
								y.add(Double.parseDouble(nextLine[2]));
								z.add(Double.parseDouble(nextLine[3]));
								gestures.add(nextLine[4]);
								nextLine = reader.readNext();
								if (nextLine != null) {
									cur_time = Double.parseDouble(nextLine[0])/1000;
								}

							}
							shift_mult++;
						}
					} else {
						init_time = time.get(0);
					}
					dyn_init = 1;

					//String[] header = "Avg Jerk X,Avg Jerk Y,Avg Jerk Z,Avg Height X,Avg Height Y,Avg Height Z,Stdev Height X,Stdev Height Y,Stdev Height Z,Avg Dist to Mean X,Avg Dist to Mean Y,Avg Dist to Mean Z,Stdev to Mean X,Stdev to Mean Y,Stdev to Mean Z,Energy X,Energy Y,Energy Z,Entropy X,Entropy Y,Entropy Z,Average X,Average Y,Average Z,Average XY,Average XZ,Average YZ,Standard Deviation X,Standard Deviation Y,Standard Deviation Z,Correlation XY,Correlation XZ,Correlation YZ,RMS X,RMS Y,RMS Z,Axis Order XY,Axis Order XZ,Axis Order YZ,Gesture".split(",");
					total_windows++;
					cur_gesture = win_gesture(gestures);
					if (time.size() >= (0.9*window_size*sampling_rate) && time.size() <= (1.1*window_size*sampling_rate)) {

						ArrayList<Double> heights_x = side_height(x, time);
						ArrayList<Double> heights_y = side_height(y, time);
						ArrayList<Double> heights_z = side_height(z, time);


						ArrayList<Double> x_peaks = peaks(x);
						ArrayList<Double> y_peaks = peaks(y);
						ArrayList<Double> z_peaks = peaks(z);

						ArrayList<Double> x_valleys = valleys(x);
						ArrayList<Double> y_valleys = valleys(y);
						ArrayList<Double> z_valleys = valleys(z);

						Double avg_peaks_x;
						Double avg_peaks_y;
						Double avg_peaks_z;
						Double stdev_peaks_x = 0.0;
						Double stdev_peaks_y = 0.0;
						Double stdev_peaks_z = 0.0;

						ArrayList<Double> sorted_x = new ArrayList<>(x);
						ArrayList<Double> sorted_y = new ArrayList<>(y);
						ArrayList<Double> sorted_z = new ArrayList<>(z);

						if (x_peaks.isEmpty()) {
							avg_peaks_x = median(sorted_x);
						} else {
							avg_peaks_x = average(x_peaks);
							stdev_peaks_x = stdev(x_peaks);
						}

						if (y_peaks.isEmpty()) {
							avg_peaks_y = median(sorted_y);
						} else {
							avg_peaks_y = average(y_peaks);
							stdev_peaks_y = stdev(y_peaks);
						}

						if (z_peaks.isEmpty()) {
							avg_peaks_z = median(sorted_z);
						} else {
							avg_peaks_z = average(z_peaks);
							stdev_peaks_z = stdev(z_peaks);
						}

						Double avg_valleys_x;
						Double avg_valleys_y;
						Double avg_valleys_z;

						Double stdev_valleys_x = 0.0;
						Double stdev_valleys_y = 0.0;
						Double stdev_valleys_z = 0.0;

						if (x_valleys.isEmpty()) {
							avg_valleys_x = median(sorted_x);
						} else {
							avg_valleys_x = average(x_valleys);
							stdev_valleys_x = stdev(x_valleys);
						}

						if (y_valleys.isEmpty()) {
							avg_valleys_y = median(sorted_y);
						} else {
							avg_valleys_y = average(y_valleys);
							stdev_valleys_y = stdev(y_valleys);
						}

						if (z_valleys.isEmpty()) {
							avg_valleys_z = median(sorted_z);
						} else {
							avg_valleys_z = average(z_valleys);
							stdev_valleys_z = stdev(z_valleys);
						}


						Double avg_height_x = 0.0;
						Double avg_height_y = 0.0;
						Double avg_height_z = 0.0;
						Double stdev_heights_x = 0.0;
						Double stdev_heights_y = 0.0;
						Double stdev_heights_z = 0.0;

						if (!heights_x.isEmpty()) {
							stdev_heights_x = stdev(heights_x);
							avg_height_x = average(heights_x);
						}

						if (!heights_y.isEmpty()) {
							stdev_heights_y = stdev(heights_y);
							avg_height_y = average(heights_y);
						}

						if (!heights_z.isEmpty()) {
							stdev_heights_z = stdev(heights_z);
							avg_height_z = average(heights_z);
						}

						String[] cur_features = (
								String.valueOf(avg_jerk(x, time, cur_length))
								+ "," + String.valueOf(avg_jerk(y, time, cur_length))
								+ "," + String.valueOf(avg_jerk(z, time, cur_length))
								+ "," + String.valueOf(avg_height_x)
								+ "," + String.valueOf(avg_height_y)
								+ "," + String.valueOf(avg_height_z)
								+ "," + String.valueOf(stdev_heights_x)
								+ "," + String.valueOf(stdev_heights_y)
								+ "," + String.valueOf(stdev_heights_z)
								+ "," + String.valueOf(energy(x))
								+ "," + String.valueOf(energy(y))
								+ "," + String.valueOf(energy(z))
								+ "," + String.valueOf(entropy(x))
								+ "," + String.valueOf(entropy(y))
								+ "," + String.valueOf(entropy(z))
								+ "," + String.valueOf(average(x)) 
								+ "," + String.valueOf(average(y)) 
								+ "," + String.valueOf(average(z))
								+ "," + String.valueOf(avg_diff(x, y, cur_length))
								+ "," + String.valueOf(avg_diff(x, z, cur_length))
								+ "," + String.valueOf(avg_diff(y, z, cur_length))
								+ "," + String.valueOf(stdev(x))
								+ "," + String.valueOf(stdev(y))
								+ "," + String.valueOf(stdev(z))
								+ "," + String.valueOf(sig_corr(x,y))
								+ "," + String.valueOf(sig_corr(x,z))
								+ "," + String.valueOf(sig_corr(y,z))
								+ "," + String.valueOf(rms(x))
								+ "," + String.valueOf(rms(y))
								+ "," + String.valueOf(rms(z))
								+ "," + String.valueOf(axis_order(x,y))
								+ "," + String.valueOf(axis_order(x,z))
								+ "," + String.valueOf(axis_order(y,z))
								+ "," + String.valueOf(x_peaks.size())
								+ "," + String.valueOf(y_peaks.size())
								+ "," + String.valueOf(z_peaks.size())
								+ "," + String.valueOf(avg_peaks_x)
								+ "," + String.valueOf(avg_peaks_y)
								+ "," + String.valueOf(avg_peaks_z)
								+ "," + String.valueOf(stdev_peaks_x)
								+ "," + String.valueOf(stdev_peaks_y)
								+ "," + String.valueOf(stdev_peaks_z)
								+ "," + String.valueOf(x_valleys.size())
								+ "," + String.valueOf(y_valleys.size())
								+ "," + String.valueOf(z_valleys.size())
								+ "," + String.valueOf(avg_valleys_x)
								+ "," + String.valueOf(avg_valleys_y)
								+ "," + String.valueOf(avg_valleys_z)
								+ "," + String.valueOf(stdev_valleys_x)
								+ "," + String.valueOf(stdev_valleys_y)
								+ "," + String.valueOf(stdev_valleys_z)
								+ "," + cur_gesture
								+ "," + String.valueOf(time.get(0))
								+ "," + String.valueOf(time.get(time.size() - 1))).split(",");

						writer.writeNext(cur_features);
					} else {
						corrupted_windows++;
						if (!cur_gesture.contains("Inactive")) {
							corrupted_gesture_windows++;
						}
					}
					total_time = 0.0;
				}
				dyn_init = 0;
				reader.close();
			}

			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return new Double[]{total_windows,corrupted_windows,corrupted_gesture_windows};
	}

	/**************************************************** 
	 ** 					Features					**
	 ****************************************************/

	/*
	 * Average Jerk
	 * Average Distance Between Axes
	 * Energy
	 * Frequency-domain entropy
	 * Mean of height of sides
	 * StdDev of height of sides
	 * Mean of distance from peak/valley to mean
	 * StdDev of distance from peak/valley to mean
	 */

	/*Find Average Jerk*/
	public static Double avg_jerk(ArrayList<Double> x, ArrayList<Double> time, int cur_length) {
		Double jerk = 0.0;
		int cnt;

		for (cnt = 1; cnt < x.size(); cnt++) {
			jerk += (((double)x.get(cnt) - (double)x.get(cnt-1)))/((time.get(cnt)) - (time.get(cnt-1)));
		}
		if (cnt == 1) {
			return (double)Math.round(jerk/cnt * 100000d) / 100000d;
		} else {
			return (double)Math.round(jerk/(cnt-1) * 100000d) / 100000d;
		}
	}

	/*Find Average Distance Between Each Value*/
	public static Double avg_diff(ArrayList<Double> x, ArrayList<Double> y, int cur_length) {
		Double diff = 0.0;
		for (int cnt = 0; cnt < x.size(); cnt++) {
			diff += x.get(cnt) - y.get(cnt);
		}
		return diff/cur_length;
	}

	/*Finds number of times axis order changes*/
	public static Integer axis_order (ArrayList<Double> x, ArrayList<Double> y) {
		int changes = 0;
		Boolean xgreatery = false;
		for (int cnt = 0; cnt < x.size(); cnt++) {
			if (cnt == 0) {
				if (x.get(cnt) > y.get(cnt)) {
					xgreatery = true;
				} else if (x.get(cnt) < y.get(cnt)){
					xgreatery = false;
				}
			} else {
				if (x.get(cnt) > y.get(cnt)) {
					if (!xgreatery) {
						changes++;
					}
				} else if (x.get(cnt) < y.get(cnt)) {
					if (xgreatery) {
						changes++;
					}
				}
			}
		}
		return changes;
	}

	/*Find Energy*/
	public static Double energy(ArrayList<Double> x) {

		int N = x.size();
		double angle;
		double energy = 0;
		for (int k = 0; k < N; k++){
			double ak = 0;
			double bk = 0;
			for (int i = 0; i < N; i++) {
				angle = 2*Math.PI*i*k/N;
				ak += x.get(i)*Math.cos(angle);
				bk += -x.get(i)*Math.sin(angle);
			}
			energy += (Math.pow(ak, 2)+Math.pow(bk, 2))/N;
		}
		return energy;

	}

	/*Find Entropy*/
	public static Double entropy(ArrayList<Double> x) {
		int N = x.size();
		double angle;
		double spectralentropy = 0;
		for (int j = 0; j < N; j++){
			double ak = 0;
			double bk = 0;
			double aj = 0;
			double bj = 0;
			double mag_j = 0;
			double mag_k = 0;
			double cj = 0;

			for (int i = 0; i < N; i++) {
				angle = 2*Math.PI*i*j/N;
				ak = x.get(i)*Math.cos(angle); //Real
				bk = -x.get(i)*Math.sin(angle); //Imaginary
				aj+=ak;
				bj+=bk;

				mag_k += Math.sqrt(Math.pow(ak, 2)+Math.pow(bk, 2));
			}
			mag_j = Math.sqrt(Math.pow(aj, 2)+Math.pow(bj, 2));

			cj = mag_j/mag_k;

			spectralentropy += cj*Math.log(cj)/Math.log(2);
		}
		return -spectralentropy;
	}

	public static Double median(ArrayList<Double> sorted_x) {
		Collections.sort(sorted_x);
		double median;
		if (sorted_x.size() % 2 == 0)
			median = (sorted_x.get(sorted_x.size()/2) + sorted_x.get(sorted_x.size()/2 - 1))/2;
		else
			median = (double) sorted_x.get(sorted_x.size()/2);
		return median;
	}

	/*calculates side_height*/
	public static ArrayList<Double> side_height(ArrayList<Double> x, ArrayList<Double> time) {
		ArrayList<Double> heights = new ArrayList<Double>();

		Boolean q1_check = false; //true greater than, false less than
		Boolean q3_check = false;
		Boolean moved_to_middle = false;
		ArrayList<Double> cur_q1_points = new ArrayList<Double>();
		ArrayList<Double> cur_q3_points = new ArrayList<Double>();
		ArrayList<Double> peaks_valleys = new ArrayList<Double>();

		ArrayList<Double> sorted_x = new ArrayList<Double>(x);
		Double median = median(sorted_x);
		Double q1 = Collections.min(x) + Math.abs((median - Collections.min(x))/2);;
		Double q3 = median + Math.abs((Collections.max(x) - median)/2);;

		Double cur_x;
		for (int i = 0; i < x.size(); i++) {
			cur_x = x.get(i);
			if (i == 0) {
				if (cur_x > q3) {
					cur_q3_points.add(cur_x);
					q1_check = true;
					q3_check = true;
				} else if (cur_x > q1) {
					q1_check = true;
				} else {
					cur_q1_points.add(cur_x);
				}
			} else {
				if (cur_x > q3) {
					q3_check = true;
					q1_check = true;
					if (moved_to_middle) {
						if (!cur_q1_points.isEmpty()) {
							peaks_valleys.add(Collections.min(cur_q1_points)); //add valley
						}
						cur_q1_points.clear();
						moved_to_middle = false;
					}
					cur_q3_points.add(cur_x);
				} else if (cur_x > q1) {
					if ((q3_check && q1_check) || (!q3_check && !q1_check)) {
						moved_to_middle = true;
					}

					q1_check = true;
					q3_check = false;
				} else {
					if (moved_to_middle) {
						if (!cur_q3_points.isEmpty()) {
							peaks_valleys.add(Collections.max(cur_q3_points)); //add peak
						}
						cur_q3_points.clear();
						moved_to_middle = false;
					}
					cur_q1_points.add(cur_x);
					q1_check = false;
					q3_check = false;
				}
			}
		}
		for (int i = 0; i < peaks_valleys.size()-1; i++) {
			heights.add(Math.abs(peaks_valleys.get(i+1) - peaks_valleys.get(i)));
		}

		return heights;
	}

	/*calculates the distance from the peak/valley to the mean*/
	public static ArrayList<Double> dist_to_mean (ArrayList<Double> x) {
		Double avg = 0.0;
		Boolean increasing = Boolean.FALSE;
		Boolean decreasing = Boolean.FALSE;
		ArrayList<Double> dist = new ArrayList<Double>();

		avg = average(x);

		for (int i = 1; i < x.size(); i++) {
			if (x.get(i) > x.get(i-1)) {
				increasing = Boolean.TRUE;
				if (decreasing) {
					dist.add(avg-x.get(i-1));
					decreasing = Boolean.FALSE;
				}
			} else if (x.get(i) < x.get(i-1)) {
				decreasing = Boolean.FALSE;
				if (increasing) {
					dist.add(x.get(i-1)-avg);
					increasing = Boolean.FALSE;
				}
			}
		}
		return dist;
	}

	/*calculates average*/
	public static Double average(ArrayList<Double> x) {
		Double avg = 0.0;
		for (int cnt = 0; cnt < x.size(); cnt++) {
			avg += x.get(cnt);
		}
		return avg/x.size();
	}

	/*Find Standard Deviation*/
	public static Double stdev(ArrayList<Double> x) {
		double avg = average(x);
		double std = 0;
		for (int i = 0; i < x.size(); i++) {
			std +=Math.pow((x.get(i) - avg),2);
		}
		return Math.sqrt(std/x.size());

	}

	/*Find Signal Correlation*/
	public static Double sig_corr(ArrayList<Double> x, ArrayList<Double> y) {
		double correlation = 0;
		int N = x.size();
		for (int cnt = 0; cnt < N; cnt++) {
			correlation += x.get(cnt) * y.get(cnt);
		}
		return correlation/N;
	}

	/*Find Root Mean Square*/
	public static Double rms(ArrayList<Double> x) {
		int N = x.size();
		Double avg = 0.0;
		for (int cnt = 0; cnt < x.size(); cnt++) {
			avg += Math.pow(x.get(cnt), 2);
		}
		return Math.sqrt(avg/N);
	}

	public static ArrayList<Double> peaks (ArrayList<Double> x) {

		ArrayList<Double> peaks = new ArrayList<Double>();

		Boolean q1_check = false; //true greater than, false less than
		Boolean q3_check = false;
		Boolean moved_to_middle = false;
		ArrayList<Double> cur_q3_points = new ArrayList<Double>();

		ArrayList<Double> sorted_x = new ArrayList<Double>(x);
		Double median = median(sorted_x);
		Double q1 = Collections.min(x) + Math.abs((median - Collections.min(x))/2);;
		Double q3 = median + Math.abs((Collections.max(x) - median)/2);;

		Double cur_x;
		for (int i = 0; i < x.size(); i++) {
			cur_x = x.get(i);
			if (i == 0) {
				if (cur_x > q3) {
					cur_q3_points.add(cur_x);
					q1_check = true;
					q3_check = true;
				} else if (cur_x > q1) {
					q1_check = true;
				}
			} else {
				if (cur_x > q3) {
					q3_check = true;
					q1_check = true;
					if (moved_to_middle) {
						moved_to_middle = false;
					}
					cur_q3_points.add(cur_x);
				} else if (cur_x > q1) {
					if ((q3_check && q1_check) || (!q3_check && !q1_check)) {
						moved_to_middle = true;
					}

					q1_check = true;
					q3_check = false;
				} else {
					if (moved_to_middle) {
						if (!cur_q3_points.isEmpty()) {
							peaks.add(Collections.max(cur_q3_points)); //add peak
						}
						cur_q3_points.clear();
						moved_to_middle = false;
					}
					q1_check = false;
					q3_check = false;
				}
			}
		}
		return peaks;
	}

	public static ArrayList<Double> valleys (ArrayList<Double> x) {

		ArrayList<Double> valleys = new ArrayList<Double>();

		Boolean q1_check = false; //true greater than, false less than
		Boolean q3_check = false;
		Boolean moved_to_middle = false;
		ArrayList<Double> cur_q1_points = new ArrayList<Double>();

		ArrayList<Double> sorted_x = new ArrayList<Double>(x);
		Double median = median(sorted_x);
		Double q1 = Collections.min(x) + Math.abs((median - Collections.min(x))/2);;
		Double q3 = median + Math.abs((Collections.max(x) - median)/2);;

		Double cur_x;
		for (int i = 0; i < x.size(); i++) {
			cur_x = x.get(i);
			if (i == 0) {
				if (cur_x > q3) {
					q1_check = true;
					q3_check = true;
				} else if (cur_x > q1) {
					q1_check = true;
				} else {
					cur_q1_points.add(cur_x);
				}
			} else {
				if (cur_x > q3) {
					q3_check = true;
					q1_check = true;
					if (moved_to_middle) {
						if (!cur_q1_points.isEmpty()) {
							valleys.add(Collections.min(cur_q1_points)); //add valley
						}
						cur_q1_points.clear();
						moved_to_middle = false;
					}
				} else if (cur_x > q1) {
					if ((q3_check && q1_check) || (!q3_check && !q1_check)) {
						moved_to_middle = true;
					}

					q1_check = true;
					q3_check = false;
				} else {
					if (moved_to_middle) {
						moved_to_middle = false;
					}
					cur_q1_points.add(cur_x);
					q1_check = false;
					q3_check = false;
				}
			}
		}
		return valleys;
	}

	/*Find Zero Crossings*/
	public static int z_crossings(ArrayList<Double> x) {
		int cur_sign;
		int prev_sign = 0; 
		int sign;
		int cnt = 0;
		int crossings = 0;
		while (prev_sign == 0 && cnt < x.size()-1) {
			prev_sign = Long.signum(x.get(cnt).longValue());
			cnt++;
		}
		if (prev_sign == 0) {
			return crossings;
		}
		while (cnt < x.size()) {
			cur_sign = Long.signum(x.get(cnt).longValue());
			while (cur_sign == 0 && cnt < x.size()-1) {
				cnt++;
				cur_sign = Long.signum(x.get(cnt).longValue());
			}
			if (cur_sign == 0) { //the last value was zero, so no more crossings will occur
				break;
			}
			sign = cur_sign - prev_sign;
			switch (sign) {
			case 2: //1-(-1)
				crossings++;
				break;
			case 0: //1-(+1), -1-(-1)
				break;
			case -2: //-1-(+1)
				crossings++;
				break;
			}
			prev_sign = cur_sign;
			cnt++;
		}

		return crossings;

	}
}

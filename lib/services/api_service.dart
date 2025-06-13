import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;

class ApiService {
  static const String baseUrl = 'http://localhost:8000';
  
  Future<Map<String, dynamic>> processImage(File imageFile) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/process-image'),
      );

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          filename: path.basename(imageFile.path),
        ),
      );

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return json.decode(responseBody);
      } else {
        throw Exception('Failed to process image: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error processing image: $e');
    }
  }

  Future<List<dynamic>> getFlashcards() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/flashcards'));
      
      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to load flashcards: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error loading flashcards: $e');
    }
  }
} 
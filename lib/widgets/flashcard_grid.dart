import 'package:flutter/material.dart';
import 'package:flutter_staggered_grid_view/flutter_staggered_grid_view.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:cached_network_image/cached_network_image.dart';

class FlashcardGrid extends StatelessWidget {
  final List<dynamic> flashcards;

  const FlashcardGrid({super.key, required this.flashcards});

  @override
  Widget build(BuildContext context) {
    if (flashcards.isEmpty) {
      return const Center(
        child: Text(
          'No flashcards yet.\nTake a picture to create one!',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 18),
        ),
      );
    }

    return MasonryGridView.count(
      padding: const EdgeInsets.all(8),
      crossAxisCount: 2,
      mainAxisSpacing: 8,
      crossAxisSpacing: 8,
      itemCount: flashcards.length,
      itemBuilder: (context, index) {
        final card = flashcards[index];
        return Card(
          elevation: 4,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (card['front']?['type'] == 'image')
                ClipRRect(
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
                  child: CachedNetworkImage(
                    imageUrl: card['front']['content'],
                    placeholder: (context, url) => const Center(
                      child: CircularProgressIndicator(),
                    ),
                    errorWidget: (context, url, error) => const Icon(Icons.error),
                  ),
                ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: MarkdownBody(
                  data: card['back']['content'] ?? '',
                  styleSheet: MarkdownStyleSheet(
                    p: const TextStyle(fontSize: 16),
                    h1: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    h2: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
              if (card['tags'] != null && (card['tags'] as List).isNotEmpty)
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Wrap(
                    spacing: 4,
                    children: (card['tags'] as List).map((tag) {
                      return Chip(
                        label: Text(tag.toString()),
                        backgroundColor: Colors.blue.shade100,
                      );
                    }).toList(),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
} 